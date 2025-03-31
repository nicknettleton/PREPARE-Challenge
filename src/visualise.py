
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import shap

import src.defaults as defaults

#------------------------------------------------------------------------------

def visualise_prediction(
    prediction,
    train_labels_path = defaults.train_labels_path,
    show = True
):
    """Displays/returns a matplotlib.pyplot with population distribution and individual prediction intervals plotted on it.

    Args:
        prediction (pd.Series): an instance from the DataFrame of predictions returned by predict(), with MAPIE intervals
        train_labels_path (string): path to the train labels file - this is used to plot the population distribution
        show (bool): Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot to be customized further after it has been created.
    
    Returns:
        matplotlib.pyplot object
    """

    sns.set_palette('dark')
    sns.set(font_scale=1.3)
    sns.set_style('white')
    
    # Load train labels to calculate risk color and display population chart
    labels = pd.read_csv(train_labels_path)

    # Risk flag
    mean = labels['composite_score'].mean()
    std = labels['composite_score'].std()

    if prediction['composite_score'] >= mean:
        risk = 'low'
    elif prediction['composite_score'] >= (mean - std):
        risk = 'medium'
    else:
        risk = 'high'
    colors = {
        'low' : ('g', 'darkgreen'),
        'medium' : ('#ed7208', '#8e4201'),
        'high' : ('r', 'darkred'),
    }
    color = colors[risk]

    # Create plot
    figure, ax = plt.subplots(1, 1)
    # plt.suptitle('Your prediction\ncompared to population', y=1)

    # Population chart
    sns.kdeplot(
        data=labels,
        x='composite_score',
        fill=True,
        ax=ax,
        alpha=0.2,
        linewidth=0,
        cut=True
    )

    # Individual chart
    y = ax.get_ylim()
    p = prediction['composite_score']
    ax.fill_betweenx(
        y, [prediction['mapie_lower_90']], [prediction['mapie_upper_90']],
        alpha=0.1,
        color=color[0]
    )
    ax.fill_betweenx(
        y, [prediction['mapie_lower_60']], [prediction['mapie_upper_60']],
        alpha=0.1,
        color=color[0]
    )
    ax.plot([p,p],y, '-', color=color[0])
    ax.annotate('Your prediction\n',
                (p,0),
                textcoords="offset points",
                xytext=(15,200),
                ha='left',
                color=color[1],
               )
    ax.annotate(str(p),
                (p,0),
                textcoords="offset points",
                xytext=(15,190),
                ha='left',
                color=color[1],
                fontsize=24
               )

    # General plot formatting
    ax.set_xlabel('Cognitive capacity')
    ax.set_ylabel('')
    ax.set_yticklabels('')
    ax.legend([
        'Population',
        'Your prediction, with\n90% and 60% likelihood'
    ], loc='lower left', bbox_to_anchor=(0, -0.4), frameon=False, ncol=2)

    ax.set_ylim(y)
    sns.despine(ax=ax, left=True)

    if show:
        plt.show()

    return plt

#------------------------------------------------------------------------------

def visualise_decision(
    index,
    ensemble_explanation,
    subestimator_explainations,
    train_labels_path = defaults.train_labels_path,
    show = True
):
    # Plot individual's SHAP decision charts

    feature_names = defaults.feature_names
    nice_feature_names = [
        (feature_names[name][:20] + '...') if len(feature_names[name]) > 20 else feature_names[name]
            for name in ensemble_explanation.feature_names
    ]

    sns.set_style('white')

    plt.figure(figsize=(5, 10))

    shap_values = [explanation.values[index] for explanation in subestimator_explainations.values()]
    shap_values = np.append(shap_values, [ensemble_explanation.values[index]], axis=0)
    highlight = [shap_values.shape[0]-1]

    # passing base_values as a list to decision_plot is currently broken
    # as workarund, values could be adjusted on the chart for the different base (expected) values for each explainer
    # but the differences are slight (<0.2) so imperceptible on the chart
    # expecteds = [exp['explainer'].expected_value for exp in explainers]
    # print(expecteds)

    legend_labels = [
        name.upper()
        for name, explanation in subestimator_explainations.items()
    ]
    
    legend_labels = [
        name.upper() + ' (' + str(round(explanation.base_values[0] + sum(explanation.values[index]), 2)) + ')'
        for name, explanation in subestimator_explainations.items()
    ] + ['Overall prediction (' + str(round(ensemble_explanation.base_values + sum(ensemble_explanation.values[index]), 0).astype('int')) + ')']

    ax = shap.decision_plot(
        ensemble_explanation.base_values,
        shap_values,
        feature_names=nice_feature_names,
        feature_display_range=slice(-1, -21, -1),
        highlight=highlight,
        title='',
        legend_labels=legend_labels,
        legend_location="lower right",
        show=False,
        auto_size_plot=False
    )
    
    ax = plt.gca()
    ax.set_xlabel('Predicted cognitive capacity')

    if show:
        plt.show()

    return plt    

#------------------------------------------------------------------------------

if __name__ == '__main__':
    print('Please see the README and the example notebook for information on how to use the visualisation tools.')