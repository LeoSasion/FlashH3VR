"""One fixed training-gradient-derived temporal coefficient for research only."""
from scripts.research_multihead_mae import calculate as original_calculate
from scripts.research_multihead_mae import evaluate as original_evaluate
from scripts.research_multihead_mae import load_case,temporal,average,gates_for

# Ratio from verified fourteen-fit-track gradients, never from validation outputs.
WEIGHT=1.5799098797912798

def calculate(delta,data,case,metric):
    _,spatial,motion,effective,pred=original_calculate(delta,data,case,metric)
    effective=dict(effective,motion=WEIGHT*motion)
    return spatial+WEIGHT*motion,spatial,motion,effective,pred

def evaluate(delta,data,case,metric):
    spatial,motion,row=original_evaluate(delta,data,case,metric)
    row.update(joint_loss=float((spatial+WEIGHT*motion).detach()),temporal_weight=WEIGHT)
    return spatial,motion,row
