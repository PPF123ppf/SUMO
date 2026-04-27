"""快速冒烟测试：验证所有基线模型能正常跑出结果"""
import os, sys
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
os.environ['SKIP_GUI_DEMO'] = '1'

# 清理残留
for f in ['tmp_routes_baseline.rou.xml', 'tmp_routes.rou.xml', 'tripinfo.xml', 'lanechanges.xml', 'fcd.xml']:
    try: os.remove(f)
    except: pass

import game_lane_change
from baseline_comparison import run_sumo_default, run_rule_based, run_no_v2x

# 1. SUMO Default
print("="*50)
print("Testing SUMO Default...")
r1 = run_sumo_default(120, '1200_TEST')
print(f"  total={r1['total_vehicles']} delay={r1['avg_delay']}s coll={r1['collisions']}")
assert r1['total_vehicles'] > 0, "SUMO Default: no vehicles passed!"
print("  SUMO Default PASSED")

# 清理
for f in ['tripinfo.xml']:
    try: os.remove(f)
    except: pass

# 2. Rule-Based
print("="*50)
print("Testing Rule-Based...")
r2 = run_rule_based(120, '1200_TEST')
print(f"  total={r2['total_vehicles']} delay={r2['avg_delay']}s coll={r2['collisions']}")
assert r2['total_vehicles'] > 0, "Rule-Based: no vehicles passed!"
print("  Rule-Based PASSED")

# 清理
for f in ['tripinfo.xml', 'tmp_routes_baseline.rou.xml']:
    try: os.remove(f)
    except: pass

# 3. No-V2X (使用game模型，但关闭全局广播)
print("="*50)
print("Testing No-V2X...")
r3 = run_no_v2x(120, '1200_TEST')
print(f"  total={r3['total_vehicles']} delay={r3['avg_delay']}s coll={r3['collisions']}")
assert r3['total_vehicles'] > 0, "No-V2X: no vehicles passed!"
print("  No-V2X PASSED")

print("="*50)
print("ALL BASELINE MODELS TEST PASSED!")
