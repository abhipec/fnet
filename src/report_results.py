"""
Report results of all test run that followed our project structure.
Plot Box plots for development set best performance on single metric.
From best on development set select the best and report results on test set.
"""

import os
import sys
import pprint
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_results_and_cleanup(result_directory, model_directory, sort_on):
    """
    Parse a single result file.
    """
    data = pd.read_csv(result_directory + model_directory + '/data_log.csv')
    sorted_results = data.sort_values(by=sort_on, ascending=False)

#    clean_up(result_directory + model_directory + '/',
#             sorted_results.index.values[3:] + 1,
#             sorted_results.index.values[0] + 1)
    return data[sort_on].max()

def clean_up(model_directory, to_delete, best):
    """
    Delete ckpt within directory.
    """
    print('Deleting unnecessary checkpoint files in ', model_directory)
    for number in to_delete:
        if os.path.isfile(model_directory + '-' + str(number)):
            os.remove(model_directory + '-' + str(number))
            os.remove(model_directory + '-' + str(number) + '.meta')
            os.remove(model_directory + 'result_' + str(number) + '.txt')
    with open(model_directory + 'checkpoint', 'w') as file_p:
        file_p.write('model_checkpoint_path: "-' + str(best) + '"\n')
        file_p.write('all_model_checkpoint_paths: "-' + str(best) + '"\n')

def parse_test_result(result_directory, model_directory, sort_on):
    """
    Report the result.
    """
    data = pd.read_csv(result_directory + model_directory + '/data_log.csv')
    sorted_results = data.sort_values(by=sort_on, ascending=False)
    final_result = sorted_results.iloc[0].to_dict()
    final_result['model_used'] = model_directory
    return final_result

def box_plot_matplotlib(result_dictionary, name_mapping, order1, order2, order3, filename):
    """
    Final plot using matplotlib.
    # assumption either order2 and order3 both will be none or all will be not none.
    """
    final_results = {}
    for model_n in result_dictionary:
        final_results[name_mapping[model_n]] = [x[0] for x in result_dictionary[model_n]]

    group_1 = []
    for model_n in order1:
        group_1.append(final_results[model_n])
    order1_label = [x[2:] for x in order1]

    group_2 = []
    for model_n in order2:
        group_2.append(final_results[model_n])
    order2_label = [x[2:] for x in order2]

    group_3 = []
    for model_n in order3:
        group_3.append(final_results[model_n])
    order3_label = [x[2:] for x in order3]

    plt.clf()
    SIZE = 13
    plt.rc('font', size=SIZE)

    fig, axes = plt.subplots(ncols=3, sharey=True)
    fig.set_size_inches(20, 7)
    
    axes[0].boxplot(group_1,
                    widths=0.5,
                    whis='range')
    axes[0].set(xticklabels=order1_label, xlabel='Wiki dataset')
    axes[0].grid(True)
    axes[0].set_ylabel('micro-F1')
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    axes[1].boxplot(group_2,
                    widths=0.5,
                    whis='range')
    axes[1].set(xticklabels=order2_label, xlabel='OntoNotes dataset')
    axes[1].grid(True)

    axes[2].boxplot(group_3,
                    widths=0.5,
                    whis='range')
    axes[2].set(xticklabels=order3_label, xlabel='BBN dataset')
    axes[2].grid(True)

    # adjust space at bottom
    fig.subplots_adjust(left=0.05, top=0.98, right=0.98, bottom=0.08, wspace=0)

    #plt.show()
    plt.savefig(filename)


def report_test_set_result(checkpoint_directory, result_dictionary):
    """
    Report the result on test set.
    """
    final_results = {}
    for model_n in result_dictionary:
        max_model_number = max(result_dictionary[model_n], key=lambda x: x[0])[1]
        final_results[model_n] = parse_test_result(checkpoint_directory,
                                                   max_model_number,
                                                   'dev_mi_F1')
    return final_results

#pylint: disable=invalid-name
if __name__ == '__main__':
    ckpt_directory = sys.argv[1]
    dir_list = os.listdir(ckpt_directory)
    models = {}
    mapping = {
        'BBN_1' : 'B-our',
        'BBN_2' : 'B-our-NoM',
        'BBN_3' : 'B-our-AllC',
        'T_BBN_model' : 'B-tl-model',
        'tf_unnorm' : 'B-tl-feature',
        'Shimaoka_BBN': 'B-Attentive',
        'AFET_BBN': 'B-AFET',
        'OntoNotes_1' : 'O-our',
        'OntoNotes_2' : 'O-our-NoM',
        'OntoNotes_3' : 'O-our-AllC',
        'T_OntoNotes_model' : 'O-tl-model',
        'Shimaoka_OntoNotes' : 'O-Attentive',
        'AFET_OntoNotes' : 'O-AFET',
        'Wiki_1' : 'W-our',
        'Wiki_2' : 'W-our-NoM',
        'Wiki_3' : 'W-our-AllC',
        'Shimaoka_Wiki' : 'W-Attentive',
        'AFET_Wiki' : 'W-AFET',
        'tf_unnorm_OntoNotes_19' : 'O-tl-feature'
    }
    for directory in dir_list:
        model_name = directory.split('.')[0]
        if not models.get(model_name, 0):
            models[model_name] = []
        models[model_name].append(directory)
    print(models)
    results = {}
    for model in models:
        results[model] = []
        for model_number in models[model]:
            results[model].append((parse_results_and_cleanup(ckpt_directory,
                                                             model_number,
                                                             'dev_mi_F1'),
                                   model_number))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)

    for model in models:
        for model2 in models:
            if model == model2:
                continue
            if model not in ['Wiki_1', 'OntoNotes_1', 'BBN_1']:
                continue
            print('Significance testing:', model, model2)
            print(stats.ttest_ind([x[0] for x in results[model]],
                                  [x[0] for x in results[model2]],
                                  equal_var=True))



    box_plot_matplotlib(results,
                        mapping,
                        ['W-AFET', 'W-Attentive', 'W-our', 'W-our-NoM', 'W-our-AllC'],
                        ['O-AFET', 'O-Attentive', 'O-our', 'O-our-NoM', 'O-our-AllC', 'O-tl-model', 'O-tl-feature'],
                        ['B-AFET', 'B-Attentive', 'B-our', 'B-our-NoM', 'B-our-AllC', 'B-tl-model', 'B-tl-feature'],
                        'box_plot.svg')

    final_result = report_test_set_result(ckpt_directory, results)
    print('final_result')
    pp.pprint(final_result)

