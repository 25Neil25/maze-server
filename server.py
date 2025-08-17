# server.py  —— 权威服务器 (Python 3.10+)
import os, asyncio, json, random
from collections import deque
import websockets

GRID_N = 18
ROUND_TIME = 90.0
BUILD_TIME = 30.0
B_MOVE_COOLDOWN = 2.0
MONITOR_TOTAL = 5.0
TRAP_STOCK_INIT = 3
TRAP_ACTIVE_TIME = 2.0
TRAP_COOLDOWN = 3.0

TICK_HZ = 30
DT = 1.0 / TICK_HZ

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

def rotate_shape(cells, k):
    out = set(cells)
    for _ in range(k % 4):
        out = {(y, -x) for (x,y) in out}
    minx = min(x for x,_ in out); miny = min(y for _,y in out)
    return {(x-minx, y-miny) for (x,y) in out}

def bfs_reachable(walls, start, goal):
    from collections import deque
    q = deque([start]); seen={start}
    while q:
        x,y = q.popleft()
        if (x,y)==goal: return True
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<GRID_N and 0<=ny<GRID_N and (nx,ny) not in walls and (nx,ny) not in seen:
                seen.add((nx,ny)); q.append((nx,ny))
    return False

def random_inner_edge_cell():
    side = random.choice(["top","bottom","left","right"])
    if side=="top": return (random.randrange(1, GRID_N-1), 1)
    if side=="bottom": return (random.randrange(1, GRID_N-1), GRID_N-2)
    if side=="left": return (1, random.randrange(1, GRID_N-1))
    return (GRID_N-2, random.randrange(1, GRID_N-1))

class GameCore:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.phase = "BUILD"
        self.tick = 0
        self.time_left = ROUND_TIME
        self.build_time = BUILD_TIME
        self.monitor_left = MONITOR_TOTAL
        self.monitor_active = False
        self.b_move_cooldown = 0.0
        self.exit_open = False
        self.win = None

        self.shape_stock = [3,3,3,3,2,2,2,1]
        self.trap_stock = TRAP_STOCK_INIT
        self.traps = []  # {'cell':(x,y),'active':False,'timer':0.0,'cool':0.0}

        self.walls, self.start, self.end, self.keys = self.build_map()
        sx, sy = self.start
        self.ax = sx + 0.5   # A 用“格坐标”中心点
        self.ay = sy + 0.5
        self.a_keys = 0

    def build_map(self):
        walls=set()
        for x in range(GRID_N):
            walls.add((x,0)); walls.add((x,GRID_N-1))
        for y in range(GRID_N):
            walls.add((0,y)); walls.add((GRID_N-1,y))
        start = random_inner_edge_cell(); end = random_inner_edge_cell()
        while end==start: end = random_inner_edge_cell()
        for y in range(2, GRID_N-2):
            for x in range(2, GRID_N-2):
                if random.random() < 0.20:
                    walls.add((x,y))
        tries=0
        while not bfs_reachable(walls, start, end) and tries<1200:
            for _ in range(40):
                rx=random.randrange(1,GRID_N-1); ry=random.randrange(1,GRID_N-1)
                walls.discard((rx,ry))
            tries+=1
        free=[(x,y) for y in range(1,GRID_N-1) for x in range(1,GRID_N-1)
              if (x,y) not in walls and (x,y) not in (start,end)]
        random.shuffle(free); keys=free[:3]
        return walls, start, end, keys

    # ==== BUILD ====
    def shape_cells_world(self, shape_idx, rot, base_gx, base_gy):
        shape = rotate_shape(SHAPES[shape_idx], rot)
        return {(base_gx+x, base_gy+y) for (x,y) in shape}

    def valid_shape_position(self, cells):
        for (gx,gy) in cells:
            if not (0<=gx<GRID_N and 0<=gy<GRID_N): return False
            if (gx,gy) in self.walls or (gx,gy) in self.keys or (gx,gy) in (self.start,self.end): return False
            if any(t['cell']==(gx,gy) for t in self.traps): return False
        tmp=set(self.walls); tmp.update(cells)
        return bfs_reachable(tmp, self.start, self.end)

    def place_shape(self, shape_idx, rot, gx, gy):
        if self.phase!="BUILD": return False,"not build"
        if not (0<=shape_idx<8): return False,"bad shape"
        if self.shape_stock[shape_idx]<=0: return False,"no stock"
        cells=self.shape_cells_world(shape_idx, rot, gx, gy)
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

    # ==== PLAY ====
    def apply_input_A(self, keys):
        # 极简移动（你可替换为你的碰撞细节）：格坐标为单位
        spd = 3.0 * DT  # 每秒 3 格
        if keys.get("w"): self.ay -= spd
        if keys.get("s"): self.ay += spd
        if keys.get("a"): self.ax -= spd
        if keys.get("d"): self.ax += spd
        # 简单防穿墙（四舍五入检测）
        gx, gy = int(self.ax), int(self.ay)
        if (gx,gy) in self.walls:
            # 回退（最简单做法）
            if keys.get("w"): self.ay += spd
            if keys.get("s"): self.ay -= spd
            if keys.get("a"): self.ax += spd
            if keys.get("d"): self.ax -= spd

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
        if not bfs_reachable(tmp, self.start, self.end): return False,"break path"
        # 砸中 A？
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
        return bfs_reachable(self.walls, a, self.end)

    # ==== TICK ====
    def tick(self):
        self.tick += 1
        if self.win: return
        if self.phase=="BUILD":
            self.build_time -= DT
            if self.build_time<=0:
                self.phase="PLAY"; self.time_left=ROUND_TIME
        elif self.phase=="PLAY":
            self.time_left -= DT
            if self.time_left<=0: self.win="B"
        # 陷阱计时 & 监视冷却
        for t in self.traps:
            if t['active']:
                t['timer'] -= DT
                if t['timer'] <= 0:
                    t['active']=False; t['timer']=0.0
            if t['cool']>0: t['cool'] = max(0.0, t['cool']-DT)
        if self.monitor_active and self.monitor_left>0:
            self.monitor_left = max(0.0, self.monitor_left - DT)

    def to_state(self):
        return {
            "tick": self.tick,
            "phase": self.phase,
            "timeLeft": round(self.time_left,2),
            "buildTime": round(self.build_time,2),
            "monitorLeft": round(self.monitor_left,2),
            "bMoveCooldown": round(self.b_move_cooldown,2),
            "exitOpen": self.a_keys>=3,
            "win": self.win,
            "A": {"x": self.ax, "y": self.ay, "keys": self.a_keys},
            "B": {},
            "walls": sorted(list(self.walls)),
            "traps": [{"cell": t["cell"], "active": t["active"]} for t in self.traps],
            "keys": list(self.keys),
            "start": self.start,
            "end": self.end
        }

class Room:
    def __init__(self):
        self.core = GameCore()
        self.players = {}   # "A" -> ws, "B" -> ws
        self.queue = deque()

    async def handle_msg(self, role, msg):
        t = msg.get("t")
        if t=="input" and role=="A" and self.core.phase=="PLAY" and not self.core.win:
            self.core.apply_input_A(msg.get("keys", {}))
            # A捡钥匙（简化：脚下格正好有钥匙）
            a = (int(self.core.ax), int(self.core.ay))
            if a in self.core.keys:
                self.core.keys.remove(a); self.core.a_keys += 1
        elif t=="place" and role=="B":
            ok,_ = self.core.place_shape(msg["shape"], msg.get("rot",0), msg["gx"], msg["gy"])
        elif t=="trap" and role=="B":
            self.core.place_trap(*msg["cell"])
        elif t=="confirm" and role=="B":
            self.core.confirm_start()
        elif t=="drag" and role=="B":
            self.core.try_drag_wall(msg["src"], msg["dst"])
        elif t=="monitor" and role=="B":
            self.core.monitor_active = bool(msg.get("on")) and self.core.monitor_left>0
        elif t=="reset":
            self.core.reset_all()

    async def run(self):
        while True:
            while self.queue:
                role, msg = self.queue.popleft()
                await self.handle_msg(role, msg)
            self.core.tick()
            await self.broadcast({"t":"state","s": self.core.to_state()})
            await asyncio.sleep(DT)

    async def broadcast(self, msg):
        data = json.dumps(msg)
        await asyncio.gather(*[
            ws.send(data) for ws in self.players.values()
            if ws and ws.open
        ], return_exceptions=True)

rooms = {}

def get_room(room_id):
    if room_id not in rooms:
        rooms[room_id] = Room()
        rooms[room_id]._task = asyncio.create_task(rooms[room_id].run())
    return rooms[room_id]

async def handler(ws, path):
    role=None; room_id="default"
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("t")=="join":
                role = "B" if msg.get("role")=="B" else "A"
                room_id = msg.get("room","default")
                r = get_room(room_id)
                # 顶掉旧连接
                old = r.players.get(role)
                if old and old.open:
                    await old.close()
                r.players[role] = ws
                await ws.send(json.dumps({"t":"state","s": r.core.to_state()}))
            else:
                r = get_room(room_id)
                r.queue.append((role, msg))
    except websockets.ConnectionClosedOK:
        pass
    except Exception as e:
        print("ws error:", e)
    finally:
        r = rooms.get(room_id)
        if r and r.players.get(role) is ws:
            r.players[role] = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print("Server on 0.0.0.0:", port)
    asyncio.get_event_loop().run_until_complete(
        websockets.serve(handler, "0.0.0.0", port)
    )
    asyncio.get_event_loop().run_forever()