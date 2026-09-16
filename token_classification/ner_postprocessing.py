"""
Module for post processing ner outputs
"""
import re
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline


def extract_zip_code(input_string):
    zip_code_pattern = r'\b(\d{5}(?:-\d{4})?|\d{6})\b'
    match = re.search(zip_code_pattern, input_string)
    if match:
        zip_code = match.group(1)
        start_index = match.start(1)
        end_index = match.end(1)
        return zip_code, start_index, end_index
    else:
        return None, None, None


def load_new_model():
    model_name = "electral_small_24_feb_ner_finetuned"
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_fine_tuned.to(device)
    return model_fine_tuned, tokenizer


model, tokenizer = load_new_model()


def single_model_output(custom_context):
    model.eval()
    start = time.time()
    device = model.device
    tokenized_output = tokenizer(custom_context, return_offsets_mapping=True)
    offsets = tokenized_output["offset_mapping"]
    nlp = pipeline("ner",
                   model=model,
                   tokenizer=tokenizer,
                   device=device,
                   aggregation_strategy="max"  # can be changed to "first and simple too"
                   )
    ner_results = nlp(custom_context)
    zip_code, start_index, end_index = extract_zip_code(custom_context)
    end = time.time()
    total_time = end - start
    total_time_formatted = "{:.2f} ms".format(total_time * 1000)
    return post_processing(ner_results, total_time_formatted, zip_code, start_index, end_index)


def post_processing(input_data, total_time, zip_code=None, start_index_zip=None, end_index_zip=None):
    response = []
    output_dict = {}

    for ner_result in input_data:
        temp_dict = {'token': ner_result['word'], 'NERTag': ner_result['entity_group'],
                     'start_index': ner_result['start'], 'end_index': ner_result['end'],
                     'confidence_score': ner_result['score']}

        if temp_dict['NERTag'] == "LOC":
            temp_dict['NERTag'] = "Location"
        elif temp_dict['NERTag'] == "PER":
            temp_dict['NERTag'] = "Person"
        else:
            temp_dict['NERTag'] = "Organization"

        if len(temp_dict['token']) == 1:
            continue
        else:
            response.append(temp_dict)

    if zip_code:
        temp_dict = {'token': zip_code, 'NERTag': "Location", 'start_index': start_index_zip,
                     'end_index': end_index_zip, 'confidence_score': 0.9852345}
        response.append(temp_dict)

    responses = []
    responses.append({"id": "145", "entities": response})
    output_dict['total_time_taken'] = total_time
    output_dict['status'] = "success"
    output_dict['response'] = responses
    return output_dict


# some test cases to test it

test_cases = ["John Doe lives at 123 Main Street, Springfield, Illinois.",
              "The Eiffel Tower is located in Paris, France.",
              "Dr. Zhang works at 456 Elm Street, Chinatown, San Francisco, California.",
              "Mount Everest is in the Himalayas.",
              "Jane Smith resides at 789 Oak Avenue, Vancouver, British Columbia, Canada.",
              "The Taj Mahal is situated in Agra, Uttar Pradesh, India.",
              "1234 Willow Lane, Los Angeles, CA is where Sarah Johnson lives.",
              "The Great Wall of China stretches across multiple provinces in China.",
              "Mark Davis is from London, United Kingdom.", "Tokyo Tower is a landmark in Tokyo, Japan.",
              "Susan Chen lives at 567 Pine Street, San Francisco, CA.",
              "The Pyramids of Giza are located near Cairo, Egypt.",
              "123 Maple Road, Toronto, Ontario, Canada is where David Lee resides.",
              "The Colosseum is in Rome, Italy.",
              "Alex Kim lives at 890 Cedar Avenue, New York, NY.",
              "The Amazon Rainforest spans across multiple South American countries.",
              "Samantha Taylor is from Sydney, New South Wales, Australia.", "The Kremlin is in Moscow, Russia.",
              "456 Elm Street, Chinatown, San Francisco, CA is where Dr. Zhang works.",
              "The Sahara Desert covers parts of several African countries.",
              "Maria Garcia resides at 901 Birch Lane, Mexico City, Mexico.",
              "The Statue of Liberty is located in New York Harbor.",
              "1212 Palm Avenue, Miami, FL is the address of Daniel Martinez.",
              "The Louvre Museum is situated in Paris, France.",
              "123 Oak Street, Austin, Texas is where Emily Wilson lives.",
              "The Sydney Opera House is in Sydney, New South Wales, Australia.",
              "1001 Pine Road, Seattle, WA is the residence of Michael Brown.",
              "The Nile River flows through multiple countries in Africa.",
              "Anna Nguyen resides at 555 Elm Street, Los Angeles, CA.",
              "The Grand Canyon is in Arizona, United States.",
              "David Taylor is from Toronto, Ontario, Canada.", "Central Park is located in Manhattan, New York City.",
              "123 Elm Street, London, United Kingdom is where Emily Johnson lives.",
              "The Alps are a mountain range in Europe.",
              "Jessica Lee resides at 789 Birch Avenue, Vancouver, British Columbia, Canada.",
              "The Vatican City is an independent city-state within Rome, Italy.",
              "234 Maple Avenue, Sydney, New South Wales, Australia is where John Smith resides.",
              "The Sahara Desert is the largest hot desert in the world.",
              "123 Palm Road, Honolulu, Hawaii is the address of Jennifer Wong.",
              "The Amazon River is the second-longest river in the world.",
              "Kevin Brown is from Vancouver, British Columbia, Canada.",
              "The Golden Gate Bridge spans the Golden Gate Strait, connecting San Francisco and Marin County.",
              "567 Cedar Street, Tokyo, Japan is the residence of Aya Yamamoto.",
              "The Andes Mountains are the longest continental mountain range in the world.",
              "Jessica Kim resides at 789 Oak Avenue, Vancouver, British Columbia, Canada.",
              "The Hollywood Sign is located in Los Angeles, California.",
              "789 Maple Lane, Paris, France is where Pierre Dubois lives.",
              "The Sahara Desert covers an area larger than the contiguous United States.",
              "Emily Chen is from Beijing, China.",
              "456 Birch Road, Rome, Italy is the residence of Marco Rossi.",
              "The Victoria Falls are located on the border of Zambia and Zimbabwe.",
              "123 Elm Street, Montreal, Quebec, Canada is where Philippe Tremblay lives."]
