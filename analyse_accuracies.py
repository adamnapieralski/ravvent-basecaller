from pathlib import Path
from numpy import dtype
from sklearn import datasets
import ravvent_mapping_evaluator as rme
import json
import numpy as np

# me = rme.MappingEvaluator()

# BEAM_WIDTH = 1
# DATASET = "lambda"

# test_results_meta = {
#     (1, 1): {
#         "raw": {
#             "id": 36,
#             "path": "",
#             "results": ()
#         },
#         "joint": {
#             "id": 33,
#             "path": "",
#             "results": ()
#         },
#         "event": {
#             "id": 39,
#             "path": "",
#             "results": ()
#         },
#     },
#     (2, 1): {
#         "raw": {
#             "id": 37,
#             "path": "",
#             "results": ()
#         },
#         "joint": {
#             "id": 40,
#             "path": "",
#             "results": ()
#         },
#         "event": {
#             "id": 39,
#             "path": "",
#             "results": ()
#         },
#     },
#     (2, 2): {
#         "raw": {
#             "id": 36,
#             "path": "",
#             "results": ()
#         },
#         "joint": {
#             "id": 40,
#             "path": "",
#             "results": ()
#         },
#         "event": {
#             "id": 39,
#             "path": "",
#             "results": ()
#         },
#     },
#     (3, 1): {
#         "raw": {
#             "id": 37,
#             "path": "",
#             "results": ()
#         },
#         "joint": {
#             "id": 40,
#             "path": "",
#             "results": ()
#         },
#         "event": {
#             "id": 38,
#             "path": "",
#             "results": ()
#         },
#     },
#     (3, 2): {
#         "raw": {
#             "id": 36,
#             "path": "",
#             "results": ()
#         },
#         "joint": {
#             "id": 40,
#             "path": "",
#             "results": ()
#         },
#         "event": {
#             "id": 39,
#             "path": "",
#             "results": ()
#         },
#     }
# }

# if DATASET == 'lambda':
#     for k, v in test_results_meta.items():
#         for d_type in v.keys():
#             if k == (1,1) or k == (2,1):
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.mask.{d_type}.encu128.encd{k[0]}.decu128.decd{k[1]}.{test_results_meta[k][d_type]['id']}.beam{BEAM_WIDTH}.json"
#             elif k == (3,2):
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.{d_type}.{test_results_meta[k][d_type]['id']}.beam{BEAM_WIDTH}.fixed.json"
#             else:
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.{d_type}.{test_results_meta[k][d_type]['id']}.beam{BEAM_WIDTH}.json"
#             test_results_meta[k][d_type]["results"] = me.compute_total_results(test_results_meta[k][d_type]["path"])

# elif DATASET == 'ecoli':
#     for k, v in test_results_meta.items():
#         for d_type in v.keys():
#             if k == (1,1) and BEAM_WIDTH == 1:
#                 test_results_meta[k][d_type]["results"] = [0,0,0]
#                 continue
#             if k == (2,1):
#                 temp_ids = {'raw': 40, 'joint': 40, 'event': 36}
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/old/mapping_evaluator_results.snippets.test.{d_type}.{temp_ids[d_type]}.ecoli.beam{BEAM_WIDTH}.json"
#             elif k == (2,2) and d_type == 'raw':
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.{d_type}.{37}.ecoli.beam{BEAM_WIDTH}.json"
#             elif k == (3,2):
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.{d_type}.{test_results_meta[k][d_type]['id']}.ecoli.beam{BEAM_WIDTH}.fixed.json"
#             else:
#                 test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.{d_type}.{test_results_meta[k][d_type]['id']}.ecoli.beam{BEAM_WIDTH}.json"
#             test_results_meta[k][d_type]["results"] = me.compute_total_results(test_results_meta[k][d_type]["path"])


# test_results_meta = {str(k): v for k, v in test_results_meta.items()}


# with open(f"accuracy_results_all.{DATASET}.beam{BEAM_WIDTH}.json", "wt") as f:
#     json.dump(test_results_meta, f, indent=2)









results_lambda_5 = Path('accuracy_results_all.lambda.beam5.json')
results_ecoli_5 = Path('accuracy_results_all.ecoli.beam5.json')

results_lambda_1 = Path('accuracy_results_all.lambda.beam1.json')
results_ecoli_1 = Path('accuracy_results_all.ecoli.beam1.json')

with open(results_lambda_5, 'rt') as f:
    results_lambda_5 = json.load(f)

with open(results_ecoli_5, 'rt') as f:
    results_ecoli_5 = json.load(f)

with open(results_lambda_1, 'rt') as f:
    results_lambda_1 = json.load(f)

with open(results_ecoli_1, 'rt') as f:
    results_ecoli_1 = json.load(f)

def get_np_results(res_dict):
    res = []
    for d_type in ['event', 'raw', 'joint']:
        res.append([x[d_type]["results"] for x in res_dict.values()])
    return np.array(res)


results_lambda_5 = get_np_results(results_lambda_5)
results_ecoli_5 = get_np_results(results_ecoli_5)

results_lambda_1 = get_np_results(results_lambda_1)
results_ecoli_1 = get_np_results(results_ecoli_1)

# print(np.mean(results_lambda_5 - results_lambda_1, axis=1))
# print(np.std(results_lambda_5 - results_lambda_1, axis=1))

results_ecoli_1[:,0,:] = results_ecoli_5[:,0,:]
print(np.mean(results_ecoli_5 - results_ecoli_1, axis=1))
print(np.std(results_ecoli_5 - results_ecoli_1, axis=1))