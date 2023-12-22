"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai
import json
import random
import openai
import time
import requests

#这个先暂时挂起
# from utils import *

# ============================================================================
# #####################[SECTION 0: initial setting] ######################
# ============================================================================
model = "Gemini"

#load the Llmama2_model api if it is specified
Llama2_api_url = ""
if model == "Llama2":
  Llama2_api_url = "" #set the llama2 server and copy the link to here , the link example is like: 'https://spip03jtgd.execute-api.us-east-1.amazonaws.com/default/call-bloom-llm'

#load the openai API if any GPT model is specified
if model == "GPT3" or model == "ChatGPT(GPT-3.5 turbo)" or model == "GPT4":
  openai.api_key = ""

#load the Gemini_model if it is specified
Gemini_model = ""
if model == "Gemini":
  import google.generativeai as genai
  genai.configure(api_key='') #Gemini api need to be put in
  Gemini_model = genai.GenerativeModel('gemini-pro')

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

# ============================================================================
# #####################[SECTION x: general STRUCTURE] ######################
# ============================================================================

def safe_generate_general(prompt, 
                          example_output="", 
                          special_instruction="", 
                          repeat=3, 
                          fail_safe_response="error", 
                          func_validate=None, 
                          func_clean_up=None, 
                          verbose=False,
                          gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 30, 
                            "temperature": 0, "top_p": 1, "stream": False,
                            "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}):
  #以下模型按照从弱到强的顺序依次排序
  if model == "Llama2":
    return Llama2_safe_generate_response(prompt, 
                                          example_output, 
                                          special_instruction, 
                                          repeat, 
                                          fail_safe_response, 
                                          func_validate, 
                                          func_clean_up, 
                                          verbose)
  
  if model == "GPT3":
    return safe_generate_response(prompt, 
                                    example_output, 
                                    special_instruction, 
                                    repeat, 
                                    fail_safe_response, 
                                    func_validate, 
                                    func_clean_up, 
                                    verbose,
                                    gpt_parameter)
  
  if model == "ChatGPT(GPT-3.5 turbo)":
    return ChatGPT_safe_generate_response(prompt, 
                                          example_output, 
                                          special_instruction, 
                                          repeat, 
                                          fail_safe_response, 
                                          func_validate, 
                                          func_clean_up, 
                                          verbose)
  
  if model == "Gemini":
    return Gemini_safe_generate_response(prompt, 
                                          example_output, 
                                          special_instruction, 
                                          repeat, 
                                          fail_safe_response, 
                                          func_validate, 
                                          func_clean_up, 
                                          verbose)
  
  if model == "GPT4":
    return GPT4_safe_generate_response(prompt, 
                                          example_output, 
                                          special_instruction, 
                                          repeat, 
                                          fail_safe_response, 
                                          func_validate, 
                                          func_clean_up, 
                                          verbose)

# ============================================================================
# #####################[SECTION 1: Llama-2 STRUCTURE] ######################
# ============================================================================
def Llama2_request(prompt): 
  """
  Given a prompt, make a request to llama-2 server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 

    a str of GPT-3's response. 
  """
  json_body = {
   "inputs": [[{"role": "system", "content": "You are a helpful chat bot"},
               {"role": "user", "content": prompt}]],
   "parameters": {"max_new_tokens":256, "top_p":0.9, "temperature":0.6}
  }

  response_json = requests.post(Llama2_api_url, json=json_body)
  return response_json["generation"]["content"]

def Llama2_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'Llama2 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: #如果是verbose的话，就会输出所有和gpt交互的信息(例如输入、输出等)
    print ("Llama2 PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = Llama2_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1  #这个用在Llama2上可能会有bug
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False

# ============================================================================
# #####################[SECTION 2: GPT-3 STRUCTURE] ######################
# ============================================================================
def GPT_request(prompt, gpt_parameter):    #调用gpt的地方
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    response = openai.Completion.create(
                model=gpt_parameter["engine"],
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    return response.choices[0].text 
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"

def safe_generate_response(prompt, 
                            example_output,
                            special_instruction,
                            repeat=3,
                            fail_safe_response="error",
                            func_validate=None,
                            func_clean_up=None,
                            verbose=False,
                            gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 30, 
                            "temperature": 0, "top_p": 1, "stream": False,
                            "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}):
  if verbose: 
    print (prompt)  

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):     #因为这里会检查输出的内容的格式(实际上使用的检查方法好像只是检测输出内容是否为空)，所以叫它safe的
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response

# ============================================================================
# #####################[SECTION 3: ChatGPT(GPT 3.5-turbo) STRUCTURE] ######################
# ============================================================================
def ChatGPT_request(prompt): 
  """
  Given a prompt, make a request to ChatGPT on OpenAI server and returns the response. 
  ARGS:
    prompt: a str prompt 
  RETURNS: 
    a str of ChatGPT's response. 
  """
  # temp_sleep()
  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"

def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: #如果是verbose的话，就会输出所有和gpt交互的信息(例如输入、输出等)
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt):   #如果输出有效的话，则直接返回。这里的重点是如何检测输出是有效的？
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:  #如果是verbose的话，就会输出所有和gpt交互的信息(例如输入、输出等)
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response

def ChatGPT_single_request(prompt): 
  temp_sleep()

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]

# ============================================================================
# #####################[SECTION 4: gemini STRUCTURE] ######################
# ============================================================================
def Gemini_request(prompt): 
  """
  Given a prompt, make a request to Gemini server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str of Gemini's response. 
  """
  # print("wait")
  response = Gemini_model.generate_content(prompt)
  # print("end")
  return response.text

def Gemini_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'Gemini Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: #如果是verbose的话，就会输出所有和gpt交互的信息(例如输入、输出等)
    print ("Gemini PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = Gemini_request(prompt).strip()
      # print("-----------------")
      # print(curr_gpt_response)
      # print("-----------------")
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      # print(type(curr_gpt_response))
      
      if func_validate(curr_gpt_response, prompt): 
        return func_clean_up(curr_gpt_response, prompt)
      
      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


# ============================================================================
# #####################[SECTION 5: GPT-4 STRUCTURE] ######################
# ============================================================================
def GPT4_request(prompt): 
  """
  Given a prompt, make a request to GPT-4 on OpenAI server and returns the response. 
  ARGS:
    prompt: a str prompt 
  RETURNS: 
    a str of GPT4's response. 
  """
  temp_sleep()

  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"

def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: #如果是verbose的话，就会输出所有和gpt交互的信息(例如输入、输出等)
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


# ============================================================================
# #####################[SECTION extra: prompt generation STRUCTURE] ######################
# ============================================================================
def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


# ============================================================================
# #####################[SECTION extra: embedding STRUCTURE] ######################
# ============================================================================
def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']


# ============================================================================
# #####################[SECTION final: test part] ######################
# ============================================================================
if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "action_location_object_v1.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response, prompt): 
    if len(gpt_response.strip()) <= 1:
      return False
    # if len(gpt_response.strip().split(" ")) > 1: #???这两行代码看着似乎有问题，为什么要拒绝长度大于1的情况
    #   return False
    return True
  
  def __func_clean_up(gpt_response, prompt):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_general(prompt, "", "", 5, "rest", __func_validate, __func_clean_up, True, gpt_parameter)

  print(output)



