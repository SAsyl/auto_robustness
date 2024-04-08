import pandas as pd
from translate import Translator

filename = "russian_comments_from_2ch_pikabu.csv"
selected_language = "ru"

df = pd.read_csv(f'./../datasets/{filename}',
                 # header=None,
                 # names=['Class', 'Review'],
                 index_col=0)

df['toxic'] = df['toxic'].apply(int)

translated_df = pd.DataFrame({'text': df['translated'], 'toxic': df['toxic']})

translated_df.to_csv(f"./../datasets/{'2ch_pikabu_eng'}.csv", index=False)
