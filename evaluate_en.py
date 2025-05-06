import os
import argparse
import numpy as np
import pandas as pd
import json
import datetime
from api import *
from judges import judge_gpt, judge_llama3, judge_rule_based
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_llama3", default=False, help="Llama3-70b")
    parser.add_argument("--evaluate_gpt", default=True, help="GPT-4")
    parser.add_argument("--input_path",type=str, default="/data/sprrty/LM/jailbreak/output/jailbreak_v1_output/2025-03-25 21:29:43.860529-target_model=v3-api-reform_model=gpt-3.5-turbo-n_restarts=5.json")
    parser.add_argument("--n_restarts", type=int, default=5, help="Number of restarts.")
   
    


    args = parser.parse_args()
    print(args.input_path)
    if 'n_restarts=5' in args.input_path:
        args.n_restarts = 5
    elif 'n_restarts=10' in args.input_path:
        args.n_restarts = 10
    else:
        print("false n_restart")
        exit()
    with open(args.input_path) as file:
        js = json.load(file)
    target_model = js["target_model"]

    timestamp_start = datetime.datetime.now()
    jailbreak_queries = js["jailbreak_queries"]
    output_js = vars(args)
    output_js['target_model'] = target_model
    
    output_js['jailbreak_results'] = []
    llama_LLM = LLMCall("llama3-70b")
    gpt_LLM = LLMCall("gpt-4")

    print(args.evaluate_llama3)
    for query_js in jailbreak_queries:
        raw_request = query_js["raw_request"]
        jailbroken_rule_results = []
        jailbroken_llama_results = []
        jailbroken_gpt_results = []

       
        for i_restart in range(args.n_restarts):
            output_jailbreak = query_js["jailbreak_outputs"][i_restart]
            # if len(output_jailbreak) > 1000:
            #     output_jailbreak = output_jailbreak[:1000]
            if args.evaluate_llama3:
                jailbroken_llama_output = judge_llama3(llama_LLM,raw_request,output_jailbreak)
                jailbroken_llama_results.append(jailbroken_llama_output)
            if args.evaluate_gpt:
                jailbroken_gpt_output = judge_gpt(gpt_LLM,raw_request,output_jailbreak)
             
                jailbroken_gpt_results.append(jailbroken_gpt_output)
            jailbroken_rule_output = judge_rule_based(output_jailbreak)
            jailbroken_rule_results.append(jailbroken_rule_output)
        query_js["jailbroken_rule_results"] = jailbroken_rule_results
        query_js["jailbroken_llama_results"] = jailbroken_llama_results
        query_js["jailbroken_gpt_results"] = jailbroken_gpt_results

        jailbroken_js = {'raw_request': raw_request,'jailbreak_output': query_js["jailbreak_outputs"], 'jailbroken_rule_results': jailbroken_rule_results, 'jailbroken_llama_results': jailbroken_llama_results, 'jailbroken_gpt_results': jailbroken_gpt_results}
        output_js['jailbreak_results'].append(jailbroken_js)

    file_name = args.input_path.strip('.json') + '-' + str(timestamp_start) + '-evaluation_result.json'


    # if 'v1' in args.input_path:
    #     file_name = f'./output/jailbreak_v1_results/{timestamp_start}-reform_model={reform_model}-target_model={target_model}-n_restarts={args.n_restarts}.json'
    #     if not os.path.exists('./output/jailbreak_v1_results'):
    #         os.makedirs('./output/jailbreak_v1_results')
    # else:

    #     file_name = f'./output/jailbreak_results/{timestamp_start}-reform_model={reform_model}-target_model={target_model}-n_restarts={args.n_restarts}.json'
    #     if not os.path.exists('./output/jailbreak_results'):
    #         os.makedirs('./output/jailbreak_results')
    with open(file_name, "w") as file:
        json.dump(output_js, file, indent=4)

    

            

                    


          