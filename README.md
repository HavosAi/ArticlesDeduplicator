# ArticlesDeduplicator

This module helps to find duplicate articles via similarities between "title", "abstract", "year" fields. You can add the "dataset" field and decide what priorities should be used while deleting duplicate articles.

Main.py has the following arguments which you can pass:

**'--first_file'** - the path to the first or main file which you want deduplicate

**'--second_file'** - the path to the second file, which you'd like to add to the main file. This file will be deduplicated and only unique articles will be added to the first file without duplicates
**'--title_column_name'** - default value - "title", the column name which represents the title of the article. The required field.
**'--abstract_column_name'** -  default value - "abstract", the column name which represents the abstract of the article. The required field.
**'--year_column_name'** -  default value - "year", the column name which represents the publication year of the article.
**'--dataset_column_name'** -  default value - "dataset", the column name which represents the source/dataset of the article.
**'--priorities_for_datasets'** - default value - "{}", Priorities for the article source. The less the number for the priority, the higher priority is". For example, you setup priorities "{'source1':8, 'source2':4, 'source3':5}", then if you have duplicate articles from the sources: source2 and source3, then an article from source2 will be left where the article from source source3 will be deleted, because source2 priority < sourcec3 priority.
**'--filename_to_save'** - a file where the deduplicated articles will be saved")
