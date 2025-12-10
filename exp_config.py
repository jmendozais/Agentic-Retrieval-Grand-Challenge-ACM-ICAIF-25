from langchain_community.vectorstores.utils import DistanceStrategy

# Distance strategy
distance_strategy_cfgs = [
    {
        "semantic_weight": 1,
        "distance_strategy": DistanceStrategy.COSINE,
        "output_path": "output/semantic_w_doc_type_template_phrases_cos.csv"
    },
    {
        "semantic_weight": 1,
        "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT,
        "output_path": "output/semantic_w_doc_type_template_phrases_dotprod.csv"
    },
    {
        "semantic_weight": 1,
        "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "output_path": "output/semantic_w_doc_type_template_phrases_euc.csv"
    },
]

# Experiment configuration
semantic_weight_cfgs = [
    {
        "semantic_weight": 0,
        "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "output_path": "output/RobertaLarge_hybrid_sw0_w_doc_type_template_phrases_euc_wclean2.csv"
    },
    {
        "semantic_weight": 0.25,
        "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "output_path": "output/RobertaLarge_hybrid_sw0.25_w_doc_type_template_phrases_euc_wclean2.csv"
    },
    {
        "semantic_weight": 0.5,
        "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "output_path": "output/RobertaLarge_hybrid_sw0.5_w_doc_type_template_phrases_euc_wclean2.csv"
    },        {
        "semantic_weight": 0.75,
        "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "output_path": "output/RobertaLarge_hybrid_sw0.75_w_doc_type_template_phrases_euc_wclean2.csv"
    },
    {
        "semantic_weight": 1,
        "distance_strategy": DistanceStrategy.EUCLIDEAN_DISTANCE,
        "output_path": "output/RobertaLarge_hybrid_sw1_w_doc_type_template_phrases_euc_wclean2.csv"
    },
]

def get_exp_configs(exp_name):
    if exp_name == "semantic_weight":
        return semantic_weight_cfgs
    elif exp_name == "distance_strategy":
        return distance_strategy_cfgs
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")