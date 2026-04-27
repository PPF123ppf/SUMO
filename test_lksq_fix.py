"""
快速验证 LKSQ 修复效果
"""
import os, sys
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
os.environ['SKIP_GUI_DEMO'] = '1'

# 清理残留
for f in ['tmp_routes.rou.xml', 'tmp_routes_baseline.rou.xml', 'tripinfo.xml']:
    try: os.remove(f)
    except: pass

import game_lane_change as glc

# 先测 Original Game（同时博弈）
print("=" * 60)
print("1. Testing Original Game (simultaneous game)...")
r_game = glc.run_once(120, 'Game_1200_TEST')
print(f"   通过车辆: {r_game['total_vehicles']}  换道: {r_game['lc_cnt']}次  延误: {r_game['avg_delay']}s  队列: {r_game['max_queue']}辆  碰撞: {r_game['collisions']}")

# 清理
for f in ['tmp_routes.rou.xml', 'tripinfo.xml']:
    try: os.remove(f)
    except: pass

# 启用 LKSQ
orig_stack = glc.USE_STACKELBERG
orig_queue = glc.USE_QUEUE_COORDINATION
glc.USE_STACKELBERG = True
glc.USE_QUEUE_COORDINATION = True

try:
    print("=" * 60)
    print("2. Testing LKSQ (Level-k + Stackelberg + Queue Coord)...")
    r_lksq = glc.run_once(120, 'LKSQ_1200_TEST')
    print(f"   通过车辆: {r_lksq['total_vehicles']}  换道: {r_lksq['lc_cnt']}次  延误: {r_lksq['avg_delay']}s  队列: {r_lksq['max_queue']}辆  碰撞: {r_lksq['collisions']}")
    print(f"   队列协调事件: {r_lksq['queue_coord_events']}")
    print(f"   Level-k分布: {r_lksq['level_k_stats']}")
    print(f"   Stackelberg决策: {r_lksq['stackelberg_cnt']}")
    print(f"   use_lksq: {r_lksq['use_lksq']}")
finally:
    glc.USE_STACKELBERG = orig_stack
    glc.USE_QUEUE_COORDINATION = orig_queue

# 判断结果
print("=" * 60)
import sys
sys.stdout.reconfigure(encoding='utf-8')
if r_lksq['lc_cnt'] > 0:
    print("LKSQ 修复成功！LKSQ 模式下产生了换道行为。lc_cnt=%d" % r_lksq['lc_cnt'])
else:
    print("LKSQ 修复仍需调整，仍无换道。")
print("=" * 60)
