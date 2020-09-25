# ArticlesDeduplicator

This module helps to find duplicate articles via similarities between "title", "abstract", "year" fields. You can add the "dataset" field and decide what priorities should be used while deleting duplicate articles.

Main.py has the following arguments which you can pass:

**'--first_file'** - the path to the first or main file which you want deduplicate. Only .xlsx/.csv files are alllowed.

**'--second_file'** - the path to the second file, which you'd like to add to the main file. This file will be deduplicated and only unique articles will be added to the first file without duplicates.  Only .xlsx/.csv files are alllowed.

**'--title_column_name'** - default value - "title", the column name which represents the title of the article. The required field.

**'--abstract_column_name'** -  default value - "abstract", the column name which represents the abstract of the article. The required field.

**'--year_column_name'** -  default value - "year", the column name which represents the publication year of the article.

**'--dataset_column_name'** -  default value - "dataset", the column name which represents the source/dataset of the article.

**'--priorities_for_datasets'** - default value - "{}", Priorities for the article source. The less the number for the priority, the higher priority is". For example, you setup priorities "{'source1':8, 'source2':4, 'source3':5}", then if you have duplicate articles from the sources: source2 and source3, then an article from source2 will be left where the article from source source3 will be deleted, because source2 priority < sourcec3 priority.

**'--filename_to_save'** - a file where the deduplicated articles will be saved. Only .xlsx/.csv files are alllowed.

## Preparing the environment
1. Please run start create_env.bat, it will create a virtual python environment.

2. Run update_env.bat, it will download necessary packages

3. Run download_packages.bat, it will download necessary resources for nltk package.

So the example of usage is

`
python main.py --first_file "file_to_process.csv" --second_file "another_file.xlsx" --priorities_for_datasets "{'source1': 2}" --filename_to_save "result.xlsx"
`
