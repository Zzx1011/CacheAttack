import json
import requests
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class SemanticTemplater:
    """
    A class to extract semantic variables from sentences, replace them one by one 
    with a substitute word, measure the semantic similarity of each resulting sentence 
    against the original, and generate a final masked template.
    """

    def __init__(self, api_key, base_url="...", model="gpt-3.5-turbo", temperature=0.0, 
                 embedding_model_name="sentence-transformers/paraphrase-albert-small-v2", 
                 similarity_threshold=0.8):
        """
        Initialize the API and Embedding Model configuration.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the API endpoint.
            model (str): The LLM model name.
            temperature (float): LLM temperature.
            embedding_model_name (str): The name of the HuggingFace embedding model.
            similarity_threshold (float): The minimum similarity required to count the substitution as 'successful'.
        """
        # API Configuration
        self.api_key = api_key
        self.api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.llm_model = model
        self.temperature = temperature
        
        # Embedding Model Configuration
        self.threshold = similarity_threshold
        
        # Determine device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model, set to evaluation mode
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embedding_model.eval()

    # Mean Pooling - Convert token embeddings to sentence embedding
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling to get a single sentence embedding.
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        # Expand attention mask to match embedding dimension
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by the number of non-padding tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Calculate Cosine Similarity
    def _calculate_similarity(self, embedding_a, embedding_b):
        """
        Calculate the Cosine Similarity between two normalized embeddings.
        """
        # Normalize vectors
        embedding_a = embedding_a / np.linalg.norm(embedding_a)
        embedding_b = embedding_b / np.linalg.norm(embedding_b)
        # Cosine Similarity is the dot product of normalized vectors
        return np.dot(embedding_a, embedding_b)

    # Generate Sentence Embedding
    def _get_sentence_embedding(self, sentence):
        """
        Tokenize a sentence and generate its normalized embedding.
        """
        # Tokenize and move to device
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings without gradient tracking
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)

        # Perform mean pooling
        sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Move back to CPU and convert to numpy array
        return sentence_embedding.cpu().numpy()[0]


    def _build_prompt(self, text):
        """
        Constructs the messages with ICL for LLM, prioritizing numerical/temporal 
        extraction to maximize successful substitutions based on high similarity.
        """
        system_prompt = (
            "You are a semantic analysis assistant. Your task is to identify 'variable entities' "
            "or 'specific details' in a sentence. These entities MUST be broken down into "
            "SINGLE WORDS. **Prioritize extracting numerical or temporal values** (e.g., times, dates, quantities) "
            "as their substitution often preserves overall sentence semantics (high similarity). "
            "For each identified original word, you must also provide a suitable 'substitute' word of the same category "
            "(e.g., Monday -> Tuesday, 7:00 -> 8:30, lamb -> beef). "
            "Return the result strictly in JSON format: "
            "{\"variables\": [{\"original\": \"word1\", \"substitute\": \"sub1\"}, {\"original\": \"word2\", \"substitute\": \"sub2\"}]}."
        )

        user_prompt_content = f"""
        Here are examples of how to extract semantic variables, prioritizing numerical/temporal changes:

        Input: "Please translate 'this is Monday' into Chinese"
        Output: {{"variables": [{{"original": "Monday", "substitute": "Tuesday"}}, {{"original": "Chinese", "substitute": "French"}}]}}

        Input: "Book a flight to Paris for tomorrow at 4 PM"
        Output: {{"variables": [{{"original": "4", "substitute": "6"}}, {{"original": "PM", "substitute": "AM"}}, {{"original": "Paris", "substitute": "London"}}, {{"original": "tomorrow", "substitute": "today"}}]}}

        Input: "Set an alarm for 7:00 AM"
        Output: {{"variables": [{{"original": "7:00", "substitute": "8:30"}}, {{"original": "AM", "substitute": "PM"}}]}}
        
        Input: "Buy 5 kilos of apples"
        Output: {{"variables": [{{"original": "5", "substitute": "10"}}, {{"original": "kilos", "substitute": "pounds"}}, {{"original": "apples", "substitute": "bananas"}}]}}

        Input: "Play specific song: Bohemian Rhapsody"
        Output: {{"variables": [{{"original": "Bohemian", "substitute": "Stairway"}}, {{"original": "Rhapsody", "substitute": "Heaven"}}]}}

        ---
        Current Task:
        Input: "{text}"
        Output:
        """

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]
        

    def process(self, text, mask_token="{{MASK}}"):
        """
        Main method to identify keywords, perform individual substitutions, 
        and measure similarity for each case.
        """
        result_content = ""
        response = None
        try:
            # 1. LLM API Call to extract variables
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            messages = self._build_prompt(text)
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"}
            }

            print(f"DEBUG: Sending request to {self.api_url}...")
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status() 

            data_raw = response.json()
            result_content = data_raw["choices"][0]["message"]["content"]
            data = json.loads(result_content)
            variables = data.get("variables", [])
            
            # --- 2. Template Generation (Simultaneous Masking of all Keywords) ---
            template_text = text
            # Sort by length for robust replacement in template
            sorted_variables = sorted(variables, key=lambda v: len(v.get("original", "")), reverse=True)
            
            for var in sorted_variables:
                original = var.get("original")
                if original and original in text:
                    template_text = template_text.replace(original, mask_token)
            
            # --- 3. Individual Substitution and Similarity Check ---
            
            # Get original sentence embedding once
            original_embedding = self._get_sentence_embedding(text)
            
            # Store results of individual substitution
            individual_results = []
            
            # Iterate through all variables and replace them one-by-one
            for var in variables:
                original = var.get("original")
                substitute = var.get("substitute")

                if original and substitute and original in text:
                    
                    # Generate the sentence with ONLY this one substitution
                    # We use replace with count=1 to only replace the first occurrence (safest for single word substitution)
                    temp_substitute_sentence = text.replace(original, substitute, 1) 
                    
                    # Check if the replacement occurred
                    if temp_substitute_sentence != text:
                        
                        # Calculate similarity
                        substitute_embedding = self._get_sentence_embedding(temp_substitute_sentence)
                        similarity_score = self._calculate_similarity(original_embedding, substitute_embedding)
                        
                        is_successful = similarity_score >= self.threshold
                        
                        individual_results.append({
                            "original_word": original,
                            "substitute_word": substitute,
                            "substituted_sentence": temp_substitute_sentence,
                            "similarity_score": float(similarity_score),
                            "is_successful": is_successful
                        })
            
            # --- 4. Final Result Compilation ---

            return {
                "original_sentence": text,
                "final_template": template_text,
                "all_variables": variables, # List of all extracted keywords/substitutes
                "individual_substitution_results": individual_results, # New: Detailed results
                "similarity_threshold": self.threshold
            }

        except requests.exceptions.HTTPError as errh:
            print(f"Error: HTTP Error occurred: {errh}")
            try:
                error_detail = response.json()
                print(f"API Error Detail: {error_detail}")
            except:
                pass
            return {"original_sentence": text, "final_template": text, "all_variables": [], "individual_substitution_results": [], "similarity_threshold": self.threshold}
        except requests.exceptions.ConnectionError as errc:
            print(f"Error: Connecting Error occurred: {errc}")
            return {"original_sentence": text, "final_template": text, "all_variables": [], "individual_substitution_results": [], "similarity_threshold": self.threshold}
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON from response. Raw LLM output: {result_content}")
            return {"original_sentence": text, "final_template": text, "all_variables": [], "individual_substitution_results": [], "similarity_threshold": self.threshold}
        except Exception as e:
            print(f"Error processing text: {e}")
            return {"original_sentence": text, "final_template": text, "all_variables": [], "individual_substitution_results": [], "similarity_threshold": self.threshold}

def run_templater(templater: SemanticTemplater, test_sentences: list):
    """
    Runs the SemanticTemplater on a list of sentences and prints the structured results.
    """
    print("--- Starting Semantic Templating Process ---")
    for sentence in test_sentences:
        result = templater.process(sentence)
        
        # Calculate overall success rate
        successful_count = sum(1 for item in result['individual_substitution_results'] if item['is_successful'])
        total_count = len(result['individual_substitution_results'])
        
        print(f"\n{'='*20} SENTENCE PROCESSED {'='*20}")
        print(f"Original Sentence: {result['original_sentence']}")
        print(f"Final Masked Template: {result['final_template']}")
        print(f"Success Rate: {successful_count}/{total_count} (Threshold: {result['similarity_threshold']})")
        print("--- Individual Substitution Results ---")
        
        if result['individual_substitution_results']:
            for item in result['individual_substitution_results']:
                status = '✅ SUCCESS' if item['is_successful'] else '❌ FAILED'
                print(f"  > Word: '{item['original_word']}' -> '{item['substitute_word']}'")
                print(f"    Similarity Score: {item['similarity_score']:.4f} ({status})")
        else:
            print("  No keywords found or substitutions performed.")
            
        print("-" * 50)
        
# --- Usage Example ---
if __name__ == "__main__":
    # 1. Instantiate the class
    templater = SemanticTemplater(
        api_key="sk-...", 
        model="...",
        similarity_threshold=0.8
    )

    # 2. Define test sentences
    # Using sentences with explicit numbers/times to test the new prioritization
    test_sentences = [
        "Please translate 'this is Tuesday' into Chinese", # Day, Language
        "Remind me to call John at 5 PM on June 15th", # Name, Time, Date (high chance of success)
        "I need 3 pounds of flour and 2 liters of water", # Quantities (high chance of success)
        "Add milk and eggs to my shopping list" # Items (lower chance of success)
    ]

    run_templater(templater, test_sentences)
    # # 3. Process each sentence
    # print("--- Starting Semantic Templating with Numerical Prioritization ---")
    # for sentence in test_sentences:
    #     result = templater.process(sentence)
        
    #     # Calculate overall success rate
    #     successful_count = sum(1 for item in result['individual_substitution_results'] if item['is_successful'])
    #     total_count = len(result['individual_substitution_results'])
        
    #     print(f"Original Sentence: {result['original_sentence']}")
    #     print(f"Final Masked Template: {result['final_template']}")
    #     print(f"Success Rate: {successful_count}/{total_count} (Threshold: {result['similarity_threshold']})")
    #     print("\n--- Individual Substitution Results ---")
        
    #     if result['individual_substitution_results']:
    #         for item in result['individual_substitution_results']:
    #             status = '✅ SUCCESS' if item['is_successful'] else '❌ FAILED'
    #             print(f"  > Original Word: {item['original_word']} -> Substitute: {item['substitute_word']}")
    #             print(f"    Similarity Score: {item['similarity_score']:.4f} ({status})")
    #     else:
    #         print("  No substitutions performed or no keywords found.")
            
    #     print("-" * 50)