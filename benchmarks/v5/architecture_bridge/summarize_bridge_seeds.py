"""Verify independent initialization and matched recipe before freezing P2 inputs."""
import argparse,ast,json
from pathlib import Path
from benchmarks.v5.encoder_research.build_joint_task_manifest import sha


class NormalizeSeed(ast.NodeTransformer):
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument' and call.args and isinstance(call.args[0],ast.Constant) and call.args[0].value=='--seed':return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='build_bridge_image':node.keywords=[k for k in node.keywords if k.arg!='seed']
        return node


def main():
    p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    first=json.loads(a.first.read_text());second=json.loads(a.second.read_text())
    assert all(x['status']=='accepted_restored_bridge_four_arm_comparison' for x in [first,second])
    folder=Path(__file__).parent
    left=NormalizeSeed().visit(ast.parse((folder/'train_restored.py').read_text()));right=NormalizeSeed().visit(ast.parse((folder/'train_replication.py').read_text()))
    assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
    roots=[Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/bridge_restored_retrain_1101_210409'),Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/bridge_seed2_orchestration_1102_012458')]
    inits=[]
    for root in roots:
        reports=[json.loads((root/arm/'report.json').read_text()) for arm in ['A','B','C','D']]
        assert len({r['identity']['seed'] for r in reports})==1;inits.append(reports[0]['identity']['seed'])
    assert inits==[20261101,20261102]
    contrasts={}
    for key in first['contrasts']:
        contrasts[key]=dict(pass_by_seed=[x['contrasts'][key]['guards']['pass_all'] for x in [first,second]],macro_Dice_delta_by_seed=[x['contrasts'][key]['guards']['candidate_minus_reference']['remote_macro_Dice'] for x in [first,second]])
    assert all(contrasts['B-A']['pass_by_seed']) and all(v<0 for v in contrasts['D-B']['macro_Dice_delta_by_seed'])
    a.output.mkdir(exist_ok=False);result=dict(status='accepted_two_seed_bridge_synthesis',seeds=inits,recipe_same_except_init_seed=True,contrasts=contrasts,first_sha256=sha(a.first),second_sha256=sha(a.second),P2_image='frozen_phikon',P2_cell='xcell_standardized64',LEMON_replacement_proven=False,fullWSI_promoted=False,scope='two training seeds on same exposed development set; not independent external confirmation')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
