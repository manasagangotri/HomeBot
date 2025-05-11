# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa-pro/concepts/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []





from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
from rasa_sdk.events import SlotSet,EventType
import google.generativeai as genai
from textblob import TextBlob
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.spatial.distance import cosine
import os 
import random
import re
import nltk
from nltk.data import find
from nltk.tokenize import sent_tokenize
from collections import defaultdict

# Load the CSV file (Ensure the correct path)
df = pd.read_csv("Home Remedies (1).csv")
ingredients_df = pd.read_csv("ingredients_file.csv")


'''
class ActionProvideHomeRemedy(Action):
    def name(self) -> Text:
        return "action_provide_home_remedy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "").lower()
        health_issue = tracker.get_slot("health_issue")
        remedy_index = tracker.get_slot("remedy_index") or 0  # Default to 0 if None
        ingredients_df = pd.read_csv("ingredients_file.csv")

       


        # Debug: Print slot values
        print(f"Debug: Health Issue Slot = {health_issue}")
        print(f"Debug: Remedy Index Slot = {remedy_index}")
         
        # 🔹 Check if the user is asking for more remedies
        if "more" in user_message:
            if not health_issue:
                dispatcher.utter_message(text="I’m sorry, I don’t remember your health issue. Can you specify again?")
                return []
            elif health_issue:
                remedies_list = df[df["Health Issue"].str.lower() == health_issue.lower()]

                # Debug: Print filtered remedies
                print(f"Debug: Filtered Remedies for {health_issue} = {remedies_list}")
                print(f"health issue is:{health_issue}")
                # 🔹 Ensure remedies exist for the health issue
                if remedies_list.empty:
                    dispatcher.utter_message(text="I’m sorry, but I don’t have any home remedies for this issue.")
                    return []

                # 🔹 Show remedies in chunks of 2
                start_idx = remedy_index
                end_idx = start_idx + 2
                remedies_chunk = remedies_list.iloc[start_idx:end_idx]

                if remedies_chunk.empty:
                    dispatcher.utter_message(text="There are no more remedies available.")
                    return []

                
                for _, row in remedies_chunk.iterrows():
                    name_of_item = row['Name of Item']
                    remedy = row['Home Remedy']
                    health_issue = row['Health Issue']
                 
                    if pd.isna(name_of_item):
                        response = (
                            f"For addressing {health_issue}, here’s a recommended remedy:\n\n{remedy}"
                        )
                    else:
                        response = (
                            f"For addressing {health_issue}, you may consider using {name_of_item}. "
                            f"Here’s a recommended remedy:\n\n{remedy}"
                        )

                    remedy_lower = remedy.lower()

                    linked_ingredients = []

                    for idx, ing_row in ingredients_df.iterrows():
                        ingredient_name = str(ing_row['Ingredient']).lower()
                        wiki_link = ing_row['Link']

                        if pd.notna(ingredient_name) and pd.notna(wiki_link):
                            if ingredient_name in remedy_lower:
                                linked_ingredients.append(f"[{ingredient_name.capitalize()} - Know More]({wiki_link})")

                    if linked_ingredients:
                        response += "\n\n\n\n🔗 Related Ingredients:\n" + "\n".join(linked_ingredients)

                    dispatcher.utter_message(text=response)



                # 🔹 Update remedy index
                new_remedy_index = end_idx

                # 🔹 If more remedies exist, prompt the user
                if end_idx < len(remedies_list):
                    dispatcher.utter_message(text="Would you like to see more remedies? Say 'more'.")
                
                dispatcher.utter_message(text="Would you like a personalized fruit suggestion based on your health profile?")



                return [
                    SlotSet("health_issue", health_issue),  # Store health issue
                    SlotSet("remedy_index", new_remedy_index),  # Update remedy index
                ]
        else:
            # Extract the health issue from user input (only if the user is not saying "more")
            health_issues = df["Health Issue"].str.lower().tolist()
            matching_issues = [issue for issue in health_issues if issue in user_message]

        
            
            if not matching_issues:
            # Check spelling suggestions using TextBlob
                corrected_text = str(TextBlob(user_message).correct())

                if corrected_text != user_message:
                    dispatcher.utter_message(
                        text=f"Sorry, I don't have a home remedy for that. Did you mean: {corrected_text}?"
                    )
                else:
                    dispatcher.utter_message(text="Sorry, I don't have a home remedy for that.")
                
                return []

            

            # 🔹 If the user specifies a new health issue, reset the remedy index
            if health_issue != matching_issues[0]:
                remedy_index = 0

            health_issue = matching_issues[0]  # Store the matched issue

        # 🔹 Filter remedies for the health issue
        remedies_list = df[df["Health Issue"].str.lower() == health_issue]

        # Debug: Print filtered remedies
        print(f"Debug: Filtered Remedies for {health_issue} = {remedies_list}")

        # 🔹 Ensure remedies exist for the health issue
        if remedies_list.empty:
            dispatcher.utter_message(text="I’m sorry, but I don’t have any home remedies for this issue.")
            return []

        # 🔹 Show remedies in chunks of 2
        start_idx = remedy_index
        end_idx = start_idx + 2
        remedies_chunk = remedies_list.iloc[start_idx:end_idx]

        if remedies_chunk.empty:
            dispatcher.utter_message(text="There are no more remedies available.")
            return []
        count=0
        for _, row in remedies_chunk.iterrows():
            name_of_item = row['Name of Item']
            remedy = row['Home Remedy']
            health_issue = row['Health Issue']
            print(f"this is remedy {remedy}")
            if(count==0):
                if pd.isna(name_of_item):
                    response = (
                                f"For addressing {health_issue}, here’s a recommended remedy:\n\n{remedy}"
                            )
                else:
                    response = (
                                f"For addressing {health_issue}, you may consider using {name_of_item}. "
                                f"Here’s a recommended remedy:\n\n{remedy}"
                            )

                remedy_lower = remedy.lower()

                linked_ingredients = []

                for idx, ing_row in ingredients_df.iterrows():
                    ingredient_name = str(ing_row['Ingredient']).lower()
                    wiki_link = ing_row['Link']

                    if pd.notna(ingredient_name) and pd.notna(wiki_link):
                        if ingredient_name in remedy_lower:
                            linked_ingredients.append(f"[{ingredient_name.capitalize()} - Know More]({wiki_link})")

                if linked_ingredients:
                    response += "\n\n🔗 Related Ingredients:\n" + "\n".join(linked_ingredients)

                   

                count=count+1
            else:
                break



            dispatcher.utter_message(text=response)
        # 🔹 Update remedy index
        new_remedy_index = end_idx

        # 🔹 If more remedies exist, prompt the user
        if end_idx < len(remedies_list):
            dispatcher.utter_message(text="Would you like to see more remedies? Say 'more'.")

        return [
            SlotSet("health_issue", health_issue),  # Store health issue
            SlotSet("remedy_index", new_remedy_index),  # Update remedy index
        ]
    
'''

class ActionProvideHomeRemedy(Action):

    def __init__(self):
        # Load models once during action server startup
        # Embedding model for semantic similarity
        self.embedder = SentenceTransformer('sentence-transformers/gtr-t5-base')

        # Paraphrasing models via transformers pipelines
        self.paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")
        self.remedy_paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

        # Load dataset from CSV
        data_dir = 'data'
        self.data = pd.read_csv( "Home Remedies (1).csv")

        # Assuming your CSV has columns 'health_issue' and 'remedy'
        self.health_issues = self.data['Health Issue'].tolist()
        self.remedies = self.data['Home Remedy'].tolist()
        self.items = self.data['Name of Item'].tolist()

        self.issue_to_remedies = defaultdict(list)
        for issue, remedy in zip(self.health_issues, self.remedies):
            key = issue.lower().strip()
            self.issue_to_remedies[key].append(remedy)
        # Load precomputed embeddings if available
        embeddings_file = os.path.join(data_dir, "C:/Users/manas/OneDrive/Desktop/New folder/health_issues_embeddings.npy")
        if os.path.exists(embeddings_file):
            self.dataset_embeddings = np.load(embeddings_file)
        else:
            # If embeddings do not exist, you may want to compute them
            self.dataset_embeddings = self.embedder.encode(self.health_issues, convert_to_numpy=True)
            np.save(embeddings_file, self.dataset_embeddings)

    def _load_lines(self, file_path: str) -> List[str]:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]



    def name(self) -> Text:
        return "action_provide_home_remedy"
    
    def _paraphrase(self, text: str, num_return_sequences: int = 3) -> List[str]:
        """Generate paraphrases for given text"""
        # Format prompt for t5 paraphraser model if needed
        queries = [f"paraphrase: {text} </s>"]
        paraphrases = self.paraphraser(
            queries,
            max_length=256,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        print(f"these are paraphrases {paraphrases}")

        paraphrased_texts = [p['generated_text'] for sublist in paraphrases for p in sublist]
        return paraphrased_texts

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into numpy array embeddings"""
        embeddings = self.embedder.encode(texts, convert_to_numpy=True,normalize_embeddings=True)
        return embeddings

    def _find_best_match(self, user_embeddings: np.ndarray) -> int:
        """
        Given user embeddings (for paraphrased prompts), find index in dataset embeddings
        with highest similarity (lowest cosine distance)
        """
        #best_idx = -1
        #best_score = float('inf')  # We minimize cosine distance = distance, similarity = 1-distance
        """ for emb in user_embeddings:
            for idx, data_emb in enumerate(self.dataset_embeddings):
                dist = cosine(emb, data_emb)
                if dist < best_score:
                    best_score = dist
                    best_idx = idx"""
        query_embedding = np.mean(user_embeddings, axis=0)
        distances = [cosine(query_embedding, emb) for emb in self.dataset_embeddings]
    
        best_score = min(distances)
        best_idx = np.argmin(distances)

        # Optional logging for debugging
        print(f"[DEBUG] Best match index: {best_idx}, Cosine Distance: {best_score}")

        # Only accept matches with good similarity
        if best_score > 0.7:
            return None  # or handle "no confident match" elsewhere
            #best_idx = np.argmin([cosine(query_embedding, emb) for emb in self.dataset_embeddings])

        return best_idx
    
    def _paraphrase_remedy(self, remedy_text: str) -> str:
        cleaned_text = re.sub(r'\s+', ' ', remedy_text.strip())  # collapse all whitespace
        cleaned_text = cleaned_text.replace("•", "-").replace("–", "-")
        """Paraphrase the remedy text"""
        prompt = f"paraphrase: {cleaned_text } </s>"
        paraphrases = self.remedy_paraphraser(
            [prompt],
            max_length=250,  # Keep shorter to avoid cutoff
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7         
        )
      
        return paraphrases[0]['generated_text']



    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "").lower()
                    

                
        health_issue = tracker.get_slot("health_issue")
        remedy_index = tracker.get_slot("remedy_index") or 0  # Default to 0 if None
        
        paraphrased_prompts = self._paraphrase(user_message, num_return_sequences=3)

        # Step 2: Embed paraphrased prompts
        user_embeddings = self._embed_texts(paraphrased_prompts)

        # Step 3: Find best matching health issue index in dataset by similarity
        best_match_idx = self._find_best_match(user_embeddings)

        if best_match_idx is None or best_match_idx == -1 or best_match_idx >= len(self.remedies):
              # No confident semantic match found, use TextBlob for spelling correction
            corrected_text = str(TextBlob(user_message).correct())

            if corrected_text != user_message:
                dispatcher.utter_message(
                    text=f"Sorry, I couldn't find a suitable home remedy for that. Did you mean: '{corrected_text}'?"
                )
            else:
                dispatcher.utter_message(text="Sorry, I could not find a suitable home remedy for your issue.")
            return []


        
        # Step 4: Retrieve remedy text from dataset
        matched_issue = self.health_issues[best_match_idx].lower().strip()
                    
            
        remedies_list = self.issue_to_remedies.get(matched_issue, [])
        
        if not remedies_list:
            dispatcher.utter_message(text="Sorry, I couldn't find any remedies for your issue.")
            return []
        # Step 4: Retrieve remedy text from dataset

        selected_remedy = remedies_list[remedy_index % len(remedies_list)]
        item_text = self.items[best_match_idx]
        #remedy_text = self.remedies[best_match_idx]

        # Step 5: Paraphrase the remedy text for more natural response
        #paraphrased_remedy = self._paraphrase_remedy(remedy_text)
        if isinstance(item_text, str) and item_text.strip():  # check if non-empty after removing whitespace
            full_response = f"Using {item_text}, you can try this remedy: {' '.join(remedies_list)}"
        else:
            full_response = selected_remedy
        
       # paraphrased_remedy = self._paraphrase_remedy(full_response)

        # Step 5: Paraphrase the remedy text for more natural response
        

        # Step 6: Return paraphrased remedy to user
        
        remedy_lower = full_response.lower()

        linked_ingredients = []

        for idx, ing_row in ingredients_df.iterrows():
            ingredient_name = str(ing_row['Ingredient']).lower()
            wiki_link = ing_row['Link']

            if pd.notna(ingredient_name) and pd.notna(wiki_link):
                if ingredient_name in remedy_lower:
                    linked_ingredients.append(f"[{ingredient_name.capitalize()} - Know More]({wiki_link})")

        if linked_ingredients:
                    full_response += "\n\n🔗 Related Ingredients:\n" + "\n".join(linked_ingredients)

        
        dispatcher.utter_message(text=full_response,metadata={"is_remedy": True})
    
        # Optionally, you could embed paraphrased remedy for other uses here.

        return []



class ActionGenerateHealthSchedule(Action):
    def name(self) -> Text:
        return "action_generate_health_schedule"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the health issue from the slot
        health_issue = tracker.get_slot("health_issue")

        if not health_issue:
            dispatcher.utter_message(text="Please specify a health issue.")
            return []

        # Configure Gemini API
        genai.configure(api_key="AIzaSyBO3-HG-WcITn58PdpK7mMyvFQitoH00qA")  # Replace with your Gemini API key

        # Define the prompt for Gemini API
        prompt = f"Generate a 7-day schedule with time to manage or recover from {health_issue} at home using home remedies. Provide the schedule in a clear and structured format without bold words."

        # Call Gemini API
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')  # Use the Gemini Pro model
            response = model.generate_content(prompt)

            # Extract the generated schedule from the API response
            schedule = response.text.strip()
            dispatcher.utter_message(text=schedule)
        except Exception as e:
            dispatcher.utter_message(text=f"Sorry, I couldn't generate the schedule. Error: {str(e)}")

        return []


# Example data
ingredient_data = {
    "garlic": {
        "uses": "Some studies show that people who eat more garlic are less likely to get certain types of cancer (garlic supplements don’t seem to have the same effect). It also may lower blood cholesterol and blood pressure levels, but it doesn’t seem to help that much"
    },
    "peppermint":{
        "uses":"Mint has been used for hundreds of years as a health remedy. Peppermint oil might help with irritable bowel syndrome -- a long-term condition that can cause cramps, bloating, gas, diarrhea, and constipation -- and it may be good for headaches as well. More studies are needed to see how much it helps and why. People use the leaf for other conditions, too, but there’s very little evidence it helps with any of them. "
    },
    "honey":{
        "uses":"This natural sweetener may work just as well for a cough as over-the-counter medicines. That could be especially helpful for children who aren’t old enough to take those. But don’t give it to an infant or a toddler younger than 1. There’s a small risk of a rare but serious kind of food poisoning that could be dangerous for them. And while you may have heard that “local” honey can help with allergies, studies don’t back that up."
    },
    "ginger": {
        "uses": "It’s been used for thousands of years in Asian medicine to treat stomachaches, diarrhea, and nausea, and studies show that it works for nausea and vomiting. There’s some evidence that it might help with menstrual cramps, too. But it’s not necessarily good for everyone. Some people get tummy trouble, heartburn, diarrhea, and gas because of it, and it may affect how some medications work. So talk to your doctor, and use it with care."
    },
    "turmeric": {
        "uses": "This spice has been hyped as being able to help with a variety of conditions from arthritis to fatty liver. There is some early research to support this. Other claims, such as healing ulcers and helping with skin rashes after radiation are lacking proof. If you try it, don’t overdo it: High doses can cause digestive problems."
    },
    "green tea":{
        "uses":"This comforting drink does more than keep you awake and alert. It’s a great source of some powerful antioxidants that can protect your cells from damage and help you fight disease. It may even lower your odds of heart disease and certain kinds of cancers, like skin, breast, lung, and colon."
    },
    "chicken soup":{
        "uses":"Turns out, Grandma was right: Chicken soup can be good for a cold. Studies show it can ease symptoms and help you get rid of it sooner. It also curbs swelling and clears out nasal fluids."
    },
    "neti spot":{
        "uses":"You put a salt and warm water mixture in something that looks like a little teapot. Then pour it through one nostril and let it drain out the other. You have to practice a little, but once you get the hang of it, it can ease allergy or cold symptoms and may even help you get rid of a cold quicker. Just make sure you use distilled or cooled boiled water and keep your neti pot clean. "
    },
    "cinnamon":{
        "uses":"You may have heard that it can help control blood sugar for people who have prediabetes or diabetes. But there’s no evidence that it does anything for any medical condition. If you plan to try it, be careful: Cinnamon extracts can be bad for your liver in large doses"
    },
    "hot bath":{
        "uses":"It’s good for all kinds of things that affect your muscles, bones, and tendons (the tissues that connect your muscles to your bones), like arthritis, back pain, and joint pain. And warm water can help get blood flow to areas that need it, so gently stretch and work those areas while you’re in there. But don’t make it too hot, especially if you have a skin condition. The ideal temperature is between 92 and 100 F"
    }
}


class ActionIngredientUses(Action):
    def name(self) -> Text:
        return "action_ingredient_uses"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the ingredient entity
        ingredient = next(tracker.get_latest_entity_values("ingredient"), None)

        if not ingredient:
            dispatcher.utter_message(text="Please specify an ingredient.")
            return []

        # Fetch the ingredient's uses and specialty
        ingredient_info = ingredient_data.get(ingredient.lower())
        if ingredient_info:
            uses = ingredient_info.get("uses", "No information available.")
            specialty = ingredient_info.get("specialty", "No specialty information available.")
            response = f"**{ingredient.capitalize()}**:\n\n- **Uses**: {uses}\n"
        else:
            response = f"Sorry, I don't have information about {ingredient}."

        dispatcher.utter_message(text=response)
        return []


class ActionAskHealthConditions(Action):
    def name(self):
        return "action_ask_health_conditions"

    def run(self, dispatcher, tracker, domain):
        df = pd.read_csv("Symptoms2L.csv")
        condition_columns = df.drop("fruits", axis=1).columns.tolist()

        idx = int(tracker.get_slot("current_condition_idx") or 0)

        if idx < len(condition_columns):
            current_condition = condition_columns[idx]
            dispatcher.utter_message(text=f"Do you have **{current_condition}**?")
            return [
                SlotSet("current_condition", current_condition)
                # Do NOT set current_condition_idx here again
            ]
        else:
            # End of condition list
            return [SlotSet("current_condition", None)]



class ActionCollectConditionResponse(Action):
    def name(self):
        return "action_collect_condition_response"

    def run(self, dispatcher, tracker, domain):
        current_condition = tracker.get_slot("current_condition")
        idx = int(tracker.get_slot("current_condition_idx") or 0)
        user_intent = tracker.latest_message["intent"].get("name")

        conditions = tracker.get_slot("health_conditions") or []

        if user_intent == "affirm" and current_condition:
            if current_condition not in conditions:
                conditions.append(current_condition)
        
        if user_intent == "deny" and current_condition:
            return [
            SlotSet("health_conditions", conditions),
            SlotSet("current_condition_idx", idx + 1),  # Move to next condition
            SlotSet("current_condition", None)          # Clear for next round
        ]
            

        print("Current Index:", idx)
        print("User conditions so far:", conditions)

        return [
            SlotSet("health_conditions", conditions),
            SlotSet("current_condition_idx", idx + 1),  # Move to next condition
            SlotSet("current_condition", None)          # Clear for next round
        ]


class ActionShowConditionButtons(Action):
    def name(self) -> Text:
        return "action_show_condition_buttons"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Load all condition names (excluding 'fruits')
        df = pd.read_csv("Symptoms2L (1).csv")
        conditions = df.drop("fruits", axis=1).columns.tolist()

        buttons = []
        for condition in conditions:
            buttons.append({
                "title": condition,
                "payload": f"/select_condition{{\"condition\": \"{condition}\"}}"
            })

        # Add a 'done' button
        buttons.append({
            "title": "Done selecting ✅",
            "payload": "/done_selecting"
        })

        dispatcher.utter_message(
            text="Please select any health conditions you have (click multiple):",
            buttons=buttons
        )
        return []

class ActionAddSelectedCondition(Action):
    def name(self) -> Text:
        return "action_add_selected_condition"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        selected_str = tracker.get_slot("condition")
        existing_conditions = tracker.get_slot("health_conditions") or []

       # if selected not in existing_conditions:
        #    existing_conditions.append(selected)
        if selected_str:
            # ✅ Split string into list and clean up spaces
            selected_list = [s.strip().lower() for s in selected_str.split(",") if s.strip()]
            print(f'this is the list: {selected_list}')   
            # ✅ Avoid duplicates (case-insensitive)
            for cond in selected_list:
                if cond.lower() not in [ec.lower() for ec in existing_conditions]:
                    existing_conditions.append(cond)

        SlotSet("health_conditions", existing_conditions)
        SlotSet("condition", None)
        return [FollowupAction("action_handle_done_selecting")]




class ActionSuggestFruit(Action):
    def name(self) -> Text:
        return "action_suggest_fruit_for_profile"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Load model and encoder
        model = joblib.load("fruit_predictor.pkl")
        le = joblib.load("label_encoder (2).pkl")
        columns = pd.read_csv("Symptoms2L (1).csv", nrows=1).drop("fruits", axis=1).columns

        # Get selected conditions
        user_conditions = tracker.get_slot("health_conditions") or []
        input_vector = np.zeros(len(columns))

        for idx, col in enumerate(columns):
            if col.lower() in [cond.lower() for cond in user_conditions]:
                input_vector[idx] = 1

        # Predict fruit
        pred_encoded = model.predict([input_vector])[0]
        fruit = le.inverse_transform([pred_encoded])[0]

        # Get fruit use
        uses_df = pd.read_csv("uses (1).xls")
        fruit_row = uses_df[uses_df["Fruit"].str.lower() == fruit.lower()]
        use = fruit_row.iloc[0]["Uses"] if not fruit_row.empty else "It's a healthy choice based on your profile."

        dispatcher.utter_message(
            text=f"Based on your health conditions, I recommend **{fruit}**.\n\n👉 {use}"
        )

        return []


class ActionHandleDoneSelecting(Action):
    def name(self) -> Text:
        return "action_handle_done_selecting"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        conditions = tracker.get_slot("health_conditions") or []

        if not conditions:
            dispatcher.utter_message(text="You haven’t selected any health conditions.")
            return []

        dispatcher.utter_message(text=f"✅ You've selected: {', '.join(conditions)}")
        return [FollowupAction("action_suggest_fruit_for_profile")]
