# server.py
import os, asyncio, json, random, time
from collections import deque
import websockets
# ---------- 常量（沿用你的数值） ----------
GRID_N = 18
TILE = 48
ROUND_TIME = 90.0
BUILD_TIME = 30.0
MONITOR_TOTAL = 5.0
B_MOVE_COOLDOWN = 2.0
TRAP_STOCK_INIT = 3
TRAP_ACTIVE_TIME = 2.0
TRAP_COOLDOWN = 3.0
SHAPES = [
    {(0,0),(1,0),(2,0),(3,0)},
    {(0,0),(1,0),(0,1),(1,1)},
    {(0,0),(1,0),(2,0),(1,1)},
    {(0,0),(0,1),(0,2),(1,2)},
    {(1,0),(1,1),(1,2),(0,2)},
    {(1,0),(2,0),(0,1),(1,1)},
    {(0,0),(1,0),(1,1),(2,1)},
    {(0,1),(1,0),(1,1),(1,2),(2,1)},
]
TICK_HZ = 30
DT = 1.0 / TICK_HZ
PROTO_VER = 1
ROOM_TTL_SEC = 300

def rotate_shape(cells, k):
    out = set(cells)
    for _ in range(k % 4):
        out = {(y, -x) for (x,y) in out}
    minx = min(x for x,_ in out); miny = min(y for _,y in out)
    return {(x-minx, y-miny) for (x,y) in out}

from collections import deque as _dq
def bfs_path(walls, start, goal):
    q = _dq([start]); seen={start}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        x,y=q.popleft()
        if (x,y)==goal: return True
        for dx,dy in dirs:
            nx,ny=x+dx,y+dy
            if 0<=nx<GRID_N and 0<=ny<GRID_N and (nx,ny) not in walls and (nx,ny) not in seen:
                seen.add((nx,ny)); q.append((nx,ny))
    return False

def random_inner_edge_cell():
    import random
    side = random.choice(["top","bottom","left","right"])
    if side=="top": return (random.randrange(1, GRID_N-1), 1)
    if side=="bottom": return (random.randrange(1, GRID_N-1), GRID_N-2)
    if side=="left": return (1, random.randrange(1, GRID_N-1))
    return (GRID_N-2, random.randrange(1, GRID_N-1))

# --------- 核心逻辑（和你一致，新增：精确碰撞 & LOBBY） ---------
class GameCore:
    def __init__(self):
        # 初始为 LOBBY（等待双方准备）
        self.phase = "LOBBY"
        self.win = None
        self.ready = {"A": False, "B": False}
        # 其它状态延迟到 start_match() 里初始化
    def start_match(self):
        # 从 LOBBY 进入 BUILD，重置一切
        self.win=None
        self.phase="BUILD"
        self.time_left = ROUND_TIME
        self.build_time = BUILD_TIME
        self.monitor_left = MONITOR_TOTAL
        self.monitor_active = False
        self.b_move_cooldown = 0.0
        self.exit_open = False
        self.shape_idx = 0
        self.shape_rot = 0
        self.shape_stock = [3,3,3,3,2,2,2,1]
        self.trap_stock = TRAP_STOCK_INIT
        self.traps = []  # {'cell':(x,y),'active':False,'timer':0.0,'cool':0.0}
        self.walls, self.start, self.end, self.keys = self.build_map()
        sx, sy = self.start
        self.ax = sx + 0.5
        self.ay = sy + 0.5
        self.a_r = 0.35  # 半径（按“格”单位）
        self.a_keys = 0

    def build_map(self):
        import random
        walls=set()
        for x in range(GRID_N):
            walls.add((x,0)); walls.add((x,GRID_N-1))
        for y in range(GRID_N):
            walls.add((0,y)); walls.add((GRID_N-1,y))
        start = random_inner_edge_cell(); end = random_inner_edge_cell()
        while end==start: end = random_inner_edge_cell()
        for y in range(2, GRID_N-2):
            for x in range(2, GRID_N-2):
                if random.random() < 0.20: walls.add((x,y))
        tries=0
        while not bfs_path(walls, start, end) and tries<1500:
            for _ in range(40):
                rx = random.randrange(1,GRID_N-1); ry = random.randrange(1,GRID_N-1)
                walls.discard((rx,ry))
            tries+=1
        free=[(x,y) for y in range(1,GRID_N-1) for x in range(1,GRID_N-1)
              if (x,y) not in walls and (x,y) not in (start,end)]
        random.shuffle(free); keys=free[:3]
        return walls, start, end, keys

    # ------- BUILD 阶段 -------
    def shape_cells_world(self, idx, rot, gx, gy):
        return {(gx+x, gy+y) for (x,y) in rotate_shape(SHAPES[idx], rot)}
    def valid_shape_position(self, cells):
        for (gx,gy) in cells:
            if not (0<=gx<GRID_N and 0<=gy<GRID_N): return False
            if (gx,gy) in self.walls or (gx,gy) in self.keys or (gx,gy) in (self.start,self.end): return False
            if any(t['cell']==(gx,gy) for t in self.traps): return False
        tmp=set(self.walls); tmp.update(cells)
        return bfs_path(tmp, self.start, self.end)
    def place_shape(self, idx, rot, gx, gy):
        if self.phase!="BUILD": return False,"not build"
        if not (0<=idx<8): return False,"bad shape"
        if self.shape_stock[idx]<=0: return False,"no stock"
        cells = self.shape_cells_world(idx, rot, gx, gy)
        if not self.valid_shape_position(cells): return False,"invalid"
        self.walls.update(cells); self.shape_stock[idx]-=1
        return True,"ok"
    def place_trap(self, gx, gy):
        if self.phase!="BUILD": return False,"not build"
        if self.trap_stock<=0: return False,"no trap"
        cell=(gx,gy)
        if not (0<=gx<GRID_N and 0<=gy<GRID_N): return False,"oob"
        if cell in self.walls or cell in self.keys or cell in (self.start,self.end): return False,"occupied"
        if any(t['cell']==cell for t in self.traps): return False,"dup"
        self.traps.append({'cell':cell,'active':False,'timer':0.0,'cool':0.0}); self.trap_stock-=1
        return True,"ok"
    def confirm_start(self):
        if self.phase=="BUILD":
            self.phase="PLAY"; self.time_left=ROUND_TIME
            return True
        return False

    # ------- PLAY 阶段：精确圆形碰撞 -------
    def circle_rect_overlap(self, cx, cy, r, rx, ry, rw, rh):
        # 计算圆心到矩形的最近点
        nx = max(rx, min(cx, rx+rw))
        ny = max(ry, min(cy, ry+rh))
        dx = cx - nx; dy = cy - ny
        return (dx*dx + dy*dy) < (r*r + 1e-9)
    def move_axis(self, nx, ny, r, dx, dy):
        # 仅沿一个轴尝试移动（dx,dy 有且只有一个非零）
        tx = nx + dx; ty = ny + dy
        # 检查附近整数格（圆可能跨越相邻格）
        minx = int((tx - r)) - 1; maxx = int((tx + r)) + 1
        miny = int((ty - r)) - 1; maxy = int((ty + r)) + 1
        for gx in range(minx, maxx+1):
            for gy in range(miny, maxy+1):
                if (gx,gy) in self.walls:
                    if self.circle_rect_overlap(tx, ty, r, gx, gy, 1.0, 1.0):
                        # 发生碰撞：取消该轴移动
                        return nx, ny
        return tx, ty

    def apply_input_A(self, keys):
        if self.phase!="PLAY" or self.win: return
        vx = (1 if keys.get("d") else 0) - (1 if keys.get("a") else 0)
        vy = (1 if keys.get("s") else 0) - (1 if keys.get("w") else 0)
        if vx==0 and vy==0: return
        # 归一
        l = (vx*vx + vy*vy) ** 0.5
        if l>0: vx/=l; vy/=l
        speed_grid_per_sec = 150.0 / TILE
        dx = vx * speed_grid_per_sec * DT
        dy = vy * speed_grid_per_sec * DT
        # 分轴移动，圆-矩形碰撞
        self.ax, self.ay = self.move_axis(self.ax, self.ay, self.a_r, dx, 0.0)
        self.ax, self.ay = self.move_axis(self.ax, self.ay, self.a_r, 0.0, dy)

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
        # 压中 A
        if int(self.ax)==dst[0] and int(self.ay)==dst[1]:
            self.walls.discard(src); self.walls.add(dst); self.win="B"; return True,"crush"
        self.walls.discard(src); self.walls.add(dst)
        self.b_move_cooldown = B_MOVE_COOLDOWN
        if not self.player_can_reach_exit(): self.win="B"
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

    def step(self):
        if self.win or self.phase=="LOBBY": return
        if self.phase=="BUILD":
            self.build_time -= DT
            if self.build_time<=0: self.phase="PLAY"; self.time_left=ROUND_TIME
        elif self.phase=="PLAY":
            a = (int(self.ax), int(self.ay))
            if a in self.keys:
                self.keys.remove(a); self.a_keys += 1
            self.exit_open = (self.a_keys>=3)
            for t in self.traps:
                if t['active'] and t['cell']==a:
                    self.win="B"; break
            if not self.win and self.exit_open and a==self.end: self.win="A"
            if not self.win and not self.player_can_reach_exit(): self.win="B"
            self.time_left -= DT
            if self.time_left<=0 and not self.win: self.win="B"
        # 计时
        for t in self.traps:
            if t['active']:
                t['timer'] -= DT
                if t['timer']<=0: t['active']=False; t['timer']=0.0
            if t['cool']>0: t['cool']=max(0.0, t['cool']-DT)
        if self.monitor_active and self.monitor_left>0:
            self.monitor_left = max(0.0, self.monitor_left - DT)
        if self.b_move_cooldown>0:
            self.b_move_cooldown = max(0.0, self.b_move_cooldown - DT)

    def to_state(self):
        # LOBBY 也会广播 ready 状态
        base = {
            "phase": self.phase,
            "win": self.win,
            "ready": {"A": self.ready["A"], "B": self.ready["B"]},
        }
        if self.phase=="LOBBY":
            return base
        base.update({
            "timeLeft": round(self.time_left,2),
            "buildTime": round(self.build_time,2),
            "monitorLeft": round(self.monitor_left,2),
            "monitorActive": self.monitor_active,
            "bMoveCooldown": round(self.b_move_cooldown,2),
            "exitOpen": self.exit_open,
            "A": {"x": self.ax, "y": self.ay, "keys": self.a_keys},
            "walls": sorted(list(self.walls)),
            "traps": [{"cell": t["cell"], "active": t["active"]} for t in self.traps],
            "keys": list(self.keys),
            "start": self.start,
            "end": self.end,
            "shapeStock": list(self.shape_stock),
        })
        return base

# --------- 房间管理（新增 ready 流程） ---------
class Room:
    def __init__(self, rid):
        self.id = rid
        self.core = GameCore()
        self.players = {"A": None, "B": None}
        self.queue = deque()
        self.last_active = time.time()
    def empty(self): return (self.players["A"] is None) and (self.players["B"] is None)
    async def handle(self, role, msg):
        t = msg.get("t")
        c = self.core
        if t=="ready":
            c.ready[role] = bool(msg.get("on", True))
            # 双方都准备 -> 开局
            if c.phase=="LOBBY" and c.ready["A"] and c.ready["B"]:
                c.start_match()
        elif t=="input" and role=="A" and c.phase=="PLAY" and not c.win:
            c.apply_input_A(msg.get("keys", {}))
        elif t=="place" and role=="B":
            ok,reason = c.place_shape(msg["shape"], msg.get("rot",0), msg["gx"], msg["gy"])
            await self.reply(role, {"t":"ack","cmd":"place","ok":ok,"reason":reason})
        elif t=="trap" and role=="B":
            if c.phase=="BUILD":
                ok,reason = c.place_trap(*msg["cell"])
            else:
                ok,reason = c.trigger_trap(msg["cell"])
            await self.reply(role, {"t":"ack","cmd":"trap","ok":ok,"reason":reason})
        elif t=="confirm" and role=="B":
            ok = c.confirm_start()
            await self.reply(role, {"t":"ack","cmd":"confirm","ok":ok})
        elif t=="drag" and role=="B":
            ok,reason = c.try_drag_wall(msg["src"], msg["dst"])
            await self.reply(role, {"t":"ack","cmd":"drag","ok":ok,"reason":reason})
        elif t=="monitor" and role=="B":
            c.monitor_active = bool(msg.get("on")) and c.monitor_left>0
            await self.reply(role, {"t":"ack","cmd":"monitor","ok":True})
        elif t=="reset":
            # 回到 LOBBY：清空准备状态
            self.core = GameCore()
            await self.reply(role, {"t":"ack","cmd":"reset","ok":True})
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
        if ws and ws.open: await ws.send(json.dumps(payload))

rooms = {}
def get_room(rid):
    if rid not in rooms:
        r = Room(rid)
        rooms[rid] = r
        asyncio.get_event_loop().create_task(r.loop())
    return rooms[rid]

async def reaper():
    while True:
        now=time.time(); dead=[]
        for rid,r in list(rooms.items()):
            if r.empty() and now-r.last_active>ROOM_TTL_SEC: dead.append(rid)
        for rid in dead:
            rooms.pop(rid, None); print("[reaper] remove room", rid)
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
                old = room.players.get(role)
                if old and old.open: await old.close()
                room.players[role] = ws
                # 新连接先收到一次 state
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
