import copy
import os

import dotenv
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import transformers

from bertopic.representation import KeyBERTInspired, TextGeneration
from bertopic.representation._utils import truncate_document

from topic_model_videos import ExtendedTopicModel

def get_prompt():
    # System prompt describes information given to all conversations
    messages = [
        {
            'role': 'system', 
            'content':"You are a helpful, respectful and concise assistant for labeling topics."
        },
        {
            'role': 'user',
            'content': """
                I have a topic that contains the following documents:
                - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
                - Meat, but especially beef, is the word food in terms of emissions.
                - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

                The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

                Based on the information about the topic above, please create a short, concise, and generic label of this topic. The label should be maximum 6 words. Make sure you to only return the label and nothing more."""
        },
        {
            'role': 'assistant',
            'content': 'Environmental impacts of eating meat'
        },
        {
            'role': 'user',
            'content': """
                I have a topic that contains the following documents:
                [DOCUMENTS]

                The topic is described by the following keywords: '[KEYWORDS]'.

                Based on the information about the topic above, please create a short, concise, and generic label of this topic. The label should be maximum 6 words. Make sure you to only return the label and nothing more."""
        }
    ]

    return messages


class ChatTextGeneration(TextGeneration):
    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf,
        topics,
    ):
        """Extract topic representations and return a single label.

        Arguments:
            topic_model: A BERTopic model
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top 4 representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # Prepare prompt
            truncated_docs = (
                [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
                if docs is not None
                else docs
            )
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            # Extract result from generator and use that as label
            topic_description = self.model(prompt, **self.pipeline_kwargs)
            topic_description = [
                (description["generated_text"][-1]['content'], 1) for description in topic_description
            ]

            if len(topic_description) < 10:
                topic_description += [("", 0) for _ in range(10 - len(topic_description))]

            updated_topics[topic] = topic_description

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = ", ".join(list(zip(*topics[topic]))[0])

        # Use a prompt that leverages either keywords or documents in
        # a custom location
        prompt = copy.deepcopy(self.prompt)
        for i in range(len(prompt)):
            if "[KEYWORDS]" in prompt[i]['content']:
                prompt[i]['content'] = prompt[i]['content'].replace("[KEYWORDS]", keywords)
            if "[DOCUMENTS]" in prompt[i]['content']:
                to_replace = ""
                for doc in docs:
                    to_replace += f"- {doc}\n"
                prompt[i]['content'] = prompt[i]['content'].replace("[DOCUMENTS]", to_replace)

        return prompt


def main():
    data_dir_path = os.path.join('.', 'data', f'topic_model_videos')
    desc_path = os.path.join(data_dir_path, 'topic_desc.parquet.gzip')

    # load topic model
    topic_model = ExtendedTopicModel.load(os.path.join(data_dir_path))

    dotenv.load_dotenv()
    HF_TOKEN = os.environ.get('HF_TOKEN')
    text_model = transformers.pipeline("text-generation", model='microsoft/Phi-4-mini-instruct', device=torch.device('cuda'), token=HF_TOKEN, torch_dtype=torch.float16)
    representation_model = [
        KeyBERTInspired(nr_repr_docs=40, nr_candidate_words=200, top_n_words=20),
        ChatTextGeneration(text_model, prompt=get_prompt(), pipeline_kwargs={'max_new_tokens': 20})
    ]
    topic_info_df = pl.read_parquet(os.path.join(data_dir_path, 'topic_info.parquet.gzip'))
    docs_df = topic_info_df.explode('Representative_Docs')
    topic_model.update_topics(docs_df['Representative_Docs'].to_list(), topics=docs_df['Topic'].to_list(), representation_model=representation_model)

    new_topic_info_df = pl.from_pandas(topic_model.get_topic_info())
    topic_info_df = topic_info_df.drop('Representation').join(new_topic_info_df.select(['Topic', 'Representation']), on='Topic', how='left')
    topic_info_df = topic_info_df.with_columns(pl.col('Representation').list.get(0).str.strip_chars().alias('Desc'))
    topic_info_df = topic_info_df.with_columns(pl.col('Desc').str.replace('TikTok ', '').str.replace(' on TikTok', '').str.replace(' TikTok', '').str.to_titlecase().str.split('\n').list.get(0))
    topic_info_df.write_parquet(desc_path, compression='gzip')

if __name__ == '__main__':
    main()