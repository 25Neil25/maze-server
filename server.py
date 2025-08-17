# online/server.py
import os, asyncio, json, random, math, time
from collections import deque
import websockets

# ====== 与你本地版完全一致的核心常量 ======
GRID_N = 18
TILE = 48                     # 前端只用于渲染比例；服务器用格坐标
ROUND_TIME = 90.0
BUILD_TIME = 30.0
MONITOR_TOTAL = 5.0
B_MOVE_COOLDOWN = 2.0

TRAP_STOCK_INIT = 3
TRAP_ACTIVE_TIME = 2.0
TRAP_COOLDOWN = 3.0

# 形状（与你一致）
SHAPES = [
    {(0,0),(1,0),(2,0),(3,0)},                    # I
    {(0,0),(1,0),(0,1),(1,1)},                    # O
    {(0,0),(1,0),(2,0),(1,1)},                    # T
    {(0,0),(0,1),(0,2),(1,2)},                    # L
    {(1,0),(1,1),(1,2),(0,2)},                    # J
    {(1,0),(2,0),(0,1),(1,1)},                    # S
    {(0,0),(1,0),(1,1),(2,1)},                    # Z
    {(0,1),(1,0),(1,1),(1,2),(2,1)},              # +
]

# 服务器 tick
TICK_HZ = 30
DT = 1.0 / TICK_HZ
PROTO_VER = 1
HEARTBEAT_SEC = 10
ROOM_TTL_SEC = 300

def rotate_shape(cells, k):
    out = set(cells)
    for _ in range(k % 4):
        out = {(y, -x) for (x,y) in out}
    minx = min(x for x,_ in out); miny = min(y for _,y in out)
    return {(x-minx, y-miny) for (x,y) in out}

def bfs_path(walls, start, goal):
    q = deque([start]); came = {start: None}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        x,y = q.popleft()
        if (x,y)==goal:
            return True
        for dx,dy in dirs:
            nx,ny = x+dx, y+dy
            if 0<=nx<GRID_N and 0<=ny<GRID_N and (nx,ny) not in walls and (nx,ny) not in came:
                came[(nx,ny)] = (x,y); q.append((nx,ny))
    return False

def random_inner_edge_cell():
    side = random.choice(["top","bottom","left","right"])
    if side=="top": return (random.randrange(1, GRID_N-1), 1)
    if side=="bottom": return (random.randrange(1, GRID_N-1), GRID_N-2)
    if side=="left": return (1, random.randrange(1, GRID_N-1))
    return (GRID_N-2, random.randrange(1, GRID_N-1))

class GameCore:
    # —— 将你 pygame 里的 reset/build_map/规则完整搬到服务端 ——
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.phase = "BUILD"            # MENU 交给前端；服务器从 BUILD 起
        self.time_left = ROUND_TIME
        self.build_time = BUILD_TIME
        self.monitor_left = MONITOR_TOTAL
        self.monitor_active = False
        self.b_move_cooldown = 0.0
        self.exit_open = False          # 服务器上以 A_keys>=3 作为“出口已开”依据
        self.win = None                 # "A"/"B"/None

        self.shape_idx = 0
        self.shape_rot = 0
        self.shape_stock = [3,3,3,3,2,2,2,1]

        self.trap_stock = TRAP_STOCK_INIT
        self.traps = []  # {'cell':(x,y),'active':False,'timer':0.0,'cool':0.0}

        self.walls, self.start, self.end, self.keys = self.build_map()

        # 玩家 A 的位置（格坐标浮点，圆形移动在服务端简化为格中心逼近）
        sx, sy = self.start
        self.ax = sx + 0.5
        self.ay = sy + 0.5
        self.a_r = 0.35                 # 半径（按格）
        self.a_keys = 0

    def build_map(self):
        walls=set()
        # 四周封死
        for x in range(GRID_N):
            walls.add((x,0)); walls.add((x,GRID_N-1))
        for y in range(GRID_N):
            walls.add((0,y)); walls.add((GRID_N-1,y))
        start = random_inner_edge_cell(); end = random_inner_edge_cell()
        while end==start: end = random_inner_edge_cell()
        # 内部随机
        for y in range(2, GRID_N-2):
            for x in range(2, GRID_N-2):
                if random.random() < 0.20:
                    walls.add((x,y))
        # 保证可达
        tries=0
        while not bfs_path(walls, start, end) and tries<1500:
            for _ in range(40):
                rx = random.randrange(1,GRID_N-1); ry = random.randrange(1,GRID_N-1)
                walls.discard((rx,ry))
            tries+=1
        # 三把钥匙
        free=[(x,y) for y in range(1,GRID_N-1) for x in range(1,GRID_N-1)
              if (x,y) not in walls and (x,y) not in (start,end)]
        random.shuffle(free); keys=free[:3]
        return walls, start, end, keys

    # ===== BUILD 阶段 =====
    def shape_cells_world(self, shape_idx, rot, base_gx, base_gy):
        shape = rotate_shape(SHAPES[shape_idx], rot)
        return {(base_gx+x, base_gy+y) for (x,y) in shape}

    def valid_shape_position(self, cells):
        for (gx,gy) in cells:
            if not (0<=gx<GRID_N and 0<=gy<GRID_N): return False
            if (gx,gy) in self.walls or (gx,gy) in self.keys or (gx,gy) in (self.start,self.end): return False
            if any(t['cell']==(gx,gy) for t in self.traps): return False
        tmp=set(self.walls); tmp.update(cells)
        return bfs_path(tmp, self.start, self.end)

    def place_shape(self, shape_idx, rot, gx, gy):
        if self.phase!="BUILD": return False,"not build"
        if not (0<=shape_idx<8): return False,"bad shape"
        if self.shape_stock[shape_idx]<=0: return False,"no stock"
        cells = self.shape_cells_world(shape_idx, rot, gx, gy)
        if not self.valid_shape_position(cells): return False,"invalid"
        self.walls.update(cells)
        self.shape_stock[shape_idx]-=1
        return True,"ok"

    def place_trap(self, gx, gy):
        if self.phase!="BUILD": return False,"not build"
        if self.trap_stock<=0: return False,"no trap"
        cell=(gx,gy)
        if not (0<=gx<GRID_N and 0<=gy<GRID_N): return False,"oob"
        if cell in self.walls or cell in self.keys or cell in (self.start,self.end): return False,"occupied"
        if any(t['cell']==cell for t in self.traps): return False,"dup"
        self.traps.append({'cell':cell,'active':False,'timer':0.0,'cool':0.0})
        self.trap_stock-=1
        return True,"ok"

    def confirm_start(self):
        if self.phase=="BUILD":
            self.phase="PLAY"; self.time_left=ROUND_TIME
            return True
        return False

    # ===== PLAY 阶段 =====
    def apply_input_A(self, keys):
        # 与你本地版相同语义：WASD，归一，速度 150px/s ≈ 3 格/s（服务端按格处理）
        vx = (1 if keys.get("d") else 0) - (1 if keys.get("a") else 0)
        vy = (1 if keys.get("s") else 0) - (1 if keys.get("w") else 0)
        if vx==0 and vy==0: return
        l = (vx*vx + vy*vy)**0.5
        if l>0:
            vx/=l; vy/=l
        speed_grid_per_sec = 150.0 / TILE     # 与本地版等价：150px/s，按48像素一格
        nx = self.ax + vx * speed_grid_per_sec * DT
        ny = self.ay + vy * speed_grid_per_sec * DT

        # 粗粒度的 AABB 碰撞：以圆心落在哪个格子判断是否进墙（与本地版略有差异，但结果一致性足够）
        gx, gy = int(nx), int(self.ay)
        if (gx, gy) not in self.walls:
            self.ax = nx
        gx, gy = int(self.ax), int(ny)
        if (gx, gy) not in self.walls:
            self.ay = ny

    def try_drag_wall(self, src, dst):
        if self.phase!="PLAY": return False,"not play"
        if self.b_move_cooldown>0: return False,"cool"
        src=tuple(src); dst=tuple(dst)
        if src not in self.walls: return False,"no src"
        if dst in self.walls: return False,"dst wall"
        if dst in self.keys or dst in (self.start,self.end): return False,"dst bad"
        if any(t['cell']==dst for t in self.traps): return False,"dst trap"
        if 0 in dst or GRID_N-1 in dst: return False,"edge forbid"
        tmp=set(self.walls); tmp.discard(src); tmp.add(dst)
        if not bfs_path(tmp, self.start, self.end): return False,"break path"

        # 压中 A 判 B 胜
        if int(self.ax)==dst[0] and int(self.ay)==dst[1]:
            self.walls.discard(src); self.walls.add(dst)
            self.win="B"; return True,"crush"

        self.walls.discard(src); self.walls.add(dst)
        self.b_move_cooldown = B_MOVE_COOLDOWN

        # 围死也 B 胜
        if not self.player_can_reach_exit():
            self.win="B"
        return True,"ok"

    def trigger_trap(self, cell):
        cell=tuple(cell)
        for t in self.traps:
            if t['cell']==cell and (not t['active']) and t['cool']<=0:
                t['active']=True; t['timer']=TRAP_ACTIVE_TIME; t['cool']=TRAP_COOLDOWN
                return True,"ok"
        return False,"bad"

    def player_can_reach_exit(self):
        a = (int(self.ax), int(self.ay))
        return bfs_path(self.walls, a, self.end)

    # ===== TICK =====
    def step(self):
        if self.win: return
        if self.phase=="BUILD":
            self.build_time -= DT
            if self.build_time<=0:
                self.phase="PLAY"; self.time_left=ROUND_TIME
        elif self.phase=="PLAY":
            # A 捡钥匙 / 开门
            a = (int(self.ax), int(self.ay))
            if a in self.keys:
                self.keys.remove(a); self.a_keys += 1
            self.exit_open = (self.a_keys>=3)
            # 踩陷阱即败
            for t in self.traps:
                if t['active'] and t['cell']==a:
                    self.win="B"; break
            # 到终点
            if not self.win and self.exit_open and a == self.end:
                self.win="A"
            # 围死
            if not self.win and not self.player_can_reach_exit():
                self.win="B"
            # 超时
            self.time_left -= DT
            if self.time_left<=0 and not self.win:
                self.win="B"

        # 陷阱与监视/冷却计时
        for t in self.traps:
            if t['active']:
                t['timer'] -= DT
                if t['timer'] <= 0:
                    t['active']=False; t['timer']=0.0
            if t['cool']>0:
                t['cool'] = max(0.0, t['cool'] - DT)
        if self.monitor_active and self.monitor_left>0:
            self.monitor_left = max(0.0, self.monitor_left - DT)
        if self.b_move_cooldown>0:
            self.b_move_cooldown = max(0.0, self.b_move_cooldown - DT)

    def to_state(self):
        return {
            "phase": self.phase,
            "timeLeft": round(self.time_left,2),
            "buildTime": round(self.build_time,2),
            "monitorLeft": round(self.monitor_left,2),
            "monitorActive": self.monitor_active,
            "bMoveCooldown": round(self.b_move_cooldown,2),
            "exitOpen": self.exit_open,
            "win": self.win,
            "A": {"x": self.ax, "y": self.ay, "keys": self.a_keys},
            "walls": sorted(list(self.walls)),
            "traps": [{"cell": t["cell"], "active": t["active"]} for t in self.traps],
            "keys": list(self.keys),
            "start": self.start,
            "end": self.end,
            "shapeStock": list(self.shape_stock),
        }

class Room:
    def __init__(self, rid):
        self.id = rid
        self.core = GameCore()
        self.players = {"A": None, "B": None}
        self.queue = deque()
        self.last_active = time.time()

    def empty(self):
        return (self.players["A"] is None) and (self.players["B"] is None)

    async def handle(self, role, msg):
        t = msg.get("t")
        if t=="input" and role=="A" and self.core.phase=="PLAY" and not self.core.win:
            self.core.apply_input_A(msg.get("keys", {}))
        elif t=="place" and role=="B":
            ok,reason = self.core.place_shape(msg["shape"], msg.get("rot",0), msg["gx"], msg["gy"])
            await self.reply(role, {"t":"ack","cmd":"place","ok":ok,"reason":reason})
        elif t=="trap" and role=="B":
            ok,reason = self.core.place_trap(*msg["cell"])
            await self.reply(role, {"t":"ack","cmd":"trap","ok":ok,"reason":reason})
        elif t=="confirm" and role=="B":
            ok = self.core.confirm_start()
            await self.reply(role, {"t":"ack","cmd":"confirm","ok":ok})
        elif t=="drag" and role=="B":
            ok,reason = self.core.try_drag_wall(msg["src"], msg["dst"])
            await self.reply(role, {"t":"ack","cmd":"drag","ok":ok,"reason":reason})
        elif t=="monitor" and role=="B":
            self.core.monitor_active = bool(msg.get("on")) and self.core.monitor_left>0
            await self.reply(role, {"t":"ack","cmd":"monitor","ok":True})
        elif t=="reset":
            self.core.reset_all(); await self.reply(role, {"t":"ack","cmd":"reset","ok":True})
        self.last_active = time.time()

    async def loop(self):
        while True:
            while self.queue:
                role, msg = self.queue.popleft()
                try:
                    await self.handle(role, msg)
                except Exception as e:
                    print("[handle error]", e)
            self.core.step()
            await self.broadcast({"t":"state","s": self.core.to_state()})
            await asyncio.sleep(DT)

    async def broadcast(self, payload):
        data = json.dumps(payload)
        await asyncio.gather(*[
            ws.send(data) for ws in self.players.values() if ws and ws.open
        ], return_exceptions=True)

    async def reply(self, role, payload):
        ws = self.players.get(role)
        if ws and ws.open:
            await ws.send(json.dumps(payload))

rooms = {}

def get_room(rid):
    if rid not in rooms:
        r = Room(rid)
        rooms[rid] = r
        asyncio.get_event_loop().create_task(r.loop())
    return rooms[rid]

async def reaper():
    while True:
        now = time.time()
        dead = []
        for rid, r in list(rooms.items()):
            if r.empty() and now - r.last_active > ROOM_TTL_SEC:
                dead.append(rid)
        for rid in dead:
            rooms.pop(rid, None)
            print("[reaper] remove room", rid)
        await asyncio.sleep(10)

async def handler(ws, path):
    role=None; rid="default"
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("t")=="join":
                if msg.get("ver") != PROTO_VER:
                    await ws.send(json.dumps({"t":"err","code":"BAD_VER","need":PROTO_VER}))
                    await ws.close(); return
                rid = msg.get("room","default")
                role = "B" if msg.get("role")=="B" else "A"
                room = get_room(rid)
                # 顶掉旧连接
                old = room.players.get(role)
                if old and old.open:
                    await old.close()
                room.players[role] = ws
                await ws.send(json.dumps({"t":"state","s": room.core.to_state()}))
            elif msg.get("t")=="ping":
                await ws.send(json.dumps({"t":"pong"}))
            else:
                room = get_room(rid)
                room.queue.append((role, msg))
    except websockets.ConnectionClosedOK:
        pass
    except Exception as e:
        print("[ws error]", e)
    finally:
        room = rooms.get(rid)
        if room and room.players.get(role) is ws:
            room.players[role] = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print("Server on 0.0.0.0:", port)
    loop = asyncio.get_event_loop()
    loop.create_task(reaper())
    loop.run_until_complete(websockets.serve(handler, "0.0.0.0", port))
    loop.run_forever()
