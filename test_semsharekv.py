import requests
import json

# 1. Load the dataset and tool description list
with open('Updated_Datasets_with_replacements.json', 'r') as f:
    datasets = json.load(f)

with open('Tool_List_Deduped.json', 'r') as f:
    tools = json.load(f)

# Create a mapping: tool_id -> description
tool_desc = {t["tool_id"]: t["description"] for t in tools}

# 2. Generate prompt for each dataset item
def make_prompt(item):
    question = item.get("question", "")
    prompt = f"""
Task description:
{question}

Please choose the most suitable tool_id from the following list and output a JSON result:
If no match is found, output "unknown".

Format:
{{"selected_tool":"python_XXXX"}}

Available tools:
{json.dumps(tool_desc, indent=2, ensure_ascii=False)}
"""
    return prompt

# 3. Call the SemanticShareKV service for inference
def call_llm(prompt):
    resp = requests.post(
        "http://127.0.0.1:8008/v1/generate",  # Assuming the service is running locally
        json={
            "prompt": prompt,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "use_semshare": True,
            "sim_threshold": 0.75
        },
    )
    return resp.json().get("text", "")

# 4. Iterate over the dataset and let LLM generate the recommended tool
results = []
for item in datasets:
    prompt = make_prompt(item)
    llm_out = call_llm(prompt)
    
    try:
        # Parse the LLM output as JSON
        suggestion = json.loads(llm_out.strip())["selected_tool"]
    except:
        suggestion = "unknown"
    
    # Add each dataset's question, ground truth, and LLM suggested tool to the results
    results.append({
        "question": item.get("question"),
        "ground_truth": item.get("target_tool"),
        "llm_suggested_tool": suggestion
    })

# Save the results to a file
with open("tool_selection_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Output statistics
correct = 0
total = len(results)
for r in results:
    if r["llm_suggested_tool"] == r["ground_truth"]:
        correct += 1

print("Accuracy:", correct / total)
