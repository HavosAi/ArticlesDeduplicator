import pandas as pd
import duplicate_finder
import argparse
import json

def read_file(filename):
    df = pd.DataFrame()
    if filename.split(".")[-1] == "csv":
        try:
            df = pd.read_csv(filename).fillna("")
        except:
            df = pd.read_csv(filename, encoding='ISO-8859-1').fillna("")
    if filename.split(".")[-1] == "xlsx":
        df = pd.read_excel(filename).fillna("")
    if len(df) == 0:
        print("filename %s has incorrect extension"%filename)
    assert len(df) != 0
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_file', help="the first file with articles")
    parser.add_argument('--second_file', default="",
        help='the second file with articles if you want to merge its content with the first file')
    parser.add_argument('--title_column_name', default="title", help="the column with the article title")
    parser.add_argument('--abstract_column_name', default="abstract", help="the column with the article abstract")
    parser.add_argument('--year_column_name', default="year", help="the column with the article year")
    parser.add_argument('--dataset_column_name', default="dataset", help="the column with the article dataset/source name")
    parser.add_argument('--priorities_for_datasets', default="{}",
        help="priorities for the article source. The less the number for the priority, the higher priority is")
    parser.add_argument('--filename_to_save', help="a file to save the deduplicated articles")
    args = parser.parse_args()
    print("First file: %s"%args.first_file)
    print("Second file: %s"%args.second_file)
    print("Filename to save: %s"%args.filename_to_save)
    print("Column for title: %s"%args.title_column_name)
    print("Column for abstract: %s"%args.abstract_column_name)
    print("Column for year: %s"%args.year_column_name)
    print("Column for dataset/source: %s"%args.dataset_column_name)
    print("Column for priorities of dataset/sources: %s"%args.priorities_for_datasets)

    first_df = read_file(args.first_file)
    second_df = None
    if args.second_file:
        second_df = read_file(args.second_file)

    _duplicate_finder = duplicate_finder.DuplicateFinder(
        title_column=args.title_column_name, abstract_column=args.abstract_column_name,
        year_column=args.year_column_name,
        dataset_column=args.dataset_column_name,
        priorities_for_datasets=json.loads(args.priorities_for_datasets)) 
    if second_df is None:
        new_df = _duplicate_finder.remove_duplicates_in_one_df_by_title(first_df)
    else:
        new_df = _duplicate_finder.deduplicate_and_process_dataset_with_merge(first_df, second_df)
    _duplicate_finder.save_to_excel(new_df, args.filename_to_save)