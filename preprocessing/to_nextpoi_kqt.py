import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
import os
from tqdm import tqdm


def normalize_traj_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def select_retrieved_history(source_data, ranked_traj_ids, limit):
    if limit <= 0:
        return source_data.iloc[0:0].copy()

    chunks = []
    seen = set()
    for traj_id in ranked_traj_ids:
        traj_id = normalize_traj_id(traj_id)
        if traj_id in seen:
            continue
        seen.add(traj_id)
        chunk = source_data[source_data['_traj_id_norm'] == traj_id]
        if chunk.empty:
            continue
        chunks.append(chunk.sort_values('UTCTimeOffsetEpoch'))
        if sum(len(c) for c in chunks) >= limit:
            break

    if not chunks:
        return source_data.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True).head(limit)


def select_same_user_history(source_data, user, start_time, limit):
    if limit <= 0:
        return source_data.iloc[0:0].copy()
    return source_data[
        (source_data['UserId'] == user) &
        (source_data['UTCTimeOffsetEpoch'] < start_time)
    ].sort_values('UTCTimeOffsetEpoch').tail(limit)


def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])
    main_data = main_data.copy()
    main_data['_traj_id_norm'] = main_data['pseudo_session_trajectory_id'].apply(normalize_traj_id)
    if historical_data is not None:
        historical_data = historical_data.copy()
        historical_data['_traj_id_norm'] = historical_data['pseudo_session_trajectory_id'].apply(normalize_traj_id)

    # List to store the QA pairs
    qa_pairs = []
    retrieval_available = 0
    retrieval_hit = 0
    fallback_used = 0

    # Iterate over each user
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]

        # Iterate over each unique trajectory for the user based on 'pseudo_session_trajectory_id'
        for traj_id in user_data['_traj_id_norm'].unique():
            user_trajectory_data = user_data[user_data['_traj_id_norm'] == traj_id]

            # Get the start time of the current trajectory
            start_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].min()

            num_traj = user_trajectory_data.shape[0]
            top200 = []
            if kqt is not None:
                top200 = kqt.get(normalize_traj_id(traj_id), kqt.get(traj_id, []))
            if top200:
                retrieval_available += 1
                top200 = [normalize_traj_id(item) for item in top200]
                # Fetch historical data before the start of the current trajectory
                history_source = historical_data if historical_data is not None else main_data
                user_historical_data = select_retrieved_history(
                    history_source,
                    top200,
                    max(0, 200 - num_traj),
                )
                if not user_historical_data.empty:
                    retrieval_hit += 1
                else:
                    fallback_used += 1
                    user_historical_data = select_same_user_history(
                        history_source,
                        user,
                        start_time_of_current_traj,
                        max(0, 600 - num_traj),
                    )
            else:
                fallback_used += 1
                history_source = historical_data if historical_data is not None else user_data
                user_historical_data = select_same_user_history(
                    history_source,
                    user,
                    start_time_of_current_traj,
                    max(0, 600 - num_traj),
                )
            user_trajectory_data.reset_index(drop=True, inplace=True)
            # Create the question based on the current trajectory (excluding the last entry) and historical data
            question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
            for i, row in user_trajectory_data.iloc[:-1].iterrows():
                if i > 0:
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")
                else:
                    question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")
            if not user_historical_data.empty:
                if len(user_trajectory_data.iloc[:-1]) > 0:
                    question_parts.append("There is also historical data:")
                else:
                    question_parts = [f"There is historical data for user {user}:"]
                for _, row in user_historical_data.iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")

            # Create the final question string
            question = " ".join(question_parts)
            value = {'nyc': 4981, 'tky': 7833, 'ca': 9690}[args.dataset_name]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # Form the answer based on the last entry of the current trajectory
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # Append the question-answer pair to the list
            qa_pairs.append((question, answer))
    print(
        "similar_traj_stats: "
        f"retrieval_available={retrieval_available}, "
        f"retrieval_hit={retrieval_hit}, "
        f"fallback_used={fallback_used}, "
        f"qa_pairs={len(qa_pairs)}"
    )
    return qa_pairs

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process dataset names.")

    # Add an argument for the dataset name
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")

    # Parse the arguments
    args = parser.parse_args()

    # Your processing code here
    print(f"Processing dataset: {args.dataset_name}")
    path = f'../datasets/{args.dataset_name}/preprocessed/'
    # Read the data
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample_with_traj.csv')
    kqt1 = jload(f'{path}train_key_top200.json')
    kqt2 = jload(f'{path}test_key_top200.json')
    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args)
    qa_pairs_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    train_out = f'{path}train_qa_pairs_kqt.json'
    train_tmp = train_out + '.tmp'
    with open(train_tmp, 'w') as json_file:
        json.dump(qa_dict_train, json_file)
    os.replace(train_tmp, train_out)


    # Save the test QA pairs in TXT format
    test_out = f'{path}test_qa_pairs_kqt.txt'
    test_tmp = test_out + '.tmp'
    with open(test_tmp, 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')
    os.replace(test_tmp, test_out)


if __name__ == "__main__":
    main()
