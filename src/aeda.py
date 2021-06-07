# %%
__doc__ = ''
__version__ = '0.0.1'

from tex_builder import TexBuilder
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import calendar
import pandas as pd
import numpy as np
import seaborn as sns
import os
import shap
import lime
import lime.lime_tabular
from interpret import show
from interpret.data import Marginal

sns.set(style="white")
path = os.getcwd()

#%%
class AEDA():
    """AEDA is a Python library to automate multiple Exploratory Data Analysis techniques.
    """

    '''
    pip install seaborn==0.10.0.rc0
    Dendrogram
    '''

    def __init__(self, data: pd.DataFrame, main_date: str, target: list, report_name: str = "AEDA", 
        report_title: str = 'Automatic Exploratory Data Analysis') -> None:
        """Constructor method.

        Args:
            data (pd.DataFrame): DataFrame that contains the data to analyze.
            main_date (str): Column name for the main Date column. `MAY ALTER TO LIST(STR) IN THE FUTURE`
            target (list): List of column names for the target column.
            report_name (str, optional): Report name. Defaults to "AEDA".
            report_title (str, optional): Report title. Defaults to 'Automatic Exploratory Data Analysis'.
        """

        self.data = data
        self.data_original = data.copy()
        self.data_date = ''
        self.main_date = main_date
        self.target = target

        self.colors = ['c', 'm', 'k', 'r', 'g', 'm', 'lightgreen', 'lightblue']
        self.week_day_name = {0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed',
                         4:'Thu', 5:'Fri', 6:'Sat'}
        self.month_name = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                           7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

        self._doc = TexBuilder(report_name, report_title)

    def start(self) -> None:
        """Method to start the AEDA analysis.
        """

        self._doc.start()

        self._prepare()

        self._about_dataset()

        self._temporal_analysis()

        self._data_analysis()

        self._doc.build()

    def _prepare(self) -> None:
        """Method to prepare the folder tree for plots and latex file.
        """

        #Create a directory for plots
        if(os.path.exists('plots') != True):
            os.mkdir('plots')

        if(os.path.exists('latex') != True):
            os.mkdir('latex')

    def _about_dataset(self) -> None:
        """Method to get DataFrame.info() and DataFrame.describe() from dataset.
        """

        self._doc.add_section('About the Dataset')
        self._doc.add_paragraph('In this section we will analyze the composition of the dataset, its dimensions and its values.')

        self._dataset_info()

        self._dataset_describe()

        self.data = self.data_original.copy()

    def _temporal_analysis(self) -> None:
        """Method that generates charts and metrics about temporal analysis.
        """

        self._doc.add_section('Temporal Analysis')
        self._doc.add_paragraph('In this section, temporal analyzes will be displayed through charts and metrics that will help to verify data inconsistency.')

        self._format_date()
        self.data['__count__'] = 1

        function_title = 'Data Consistency'
        agg_function = 'count'
        agg_value = '__count__'
        self._temporal_analysis_consistency(function_title, agg_function, agg_value)

        function_title = 'Count Samples'
        agg_function = 'count'
        agg_value = '__count__'
        self._temporal_analysis_calculate(function_title, agg_function, agg_value)

        self._doc.add_subsection('Individual Target Analysis')
        for target in self.target:
            self._doc.add_subsubsection(target)
            function_title = 'Sum %s' % (str(target))
            agg_function = 'sum'
            agg_value = target
            self._temporal_analysis_calculate(function_title, agg_function, agg_value, issub=False)

            function_title = 'Avg %s' % (str(target))
            agg_function = 'mean'
            agg_value = target
            self._temporal_analysis_calculate(function_title, agg_function, agg_value, issub=False)

        self.data = self.data_original.copy()

    def _data_analysis(self) -> None:
        """Method that calls another data analysis methods.
        """

        self._doc.add_section('Data Analysis')

        self._data_analysis_corr()

        self._data_analysis_values()

    def _data_analysis_shap(self, target: str) -> None:
        """Method that applies the Shapley value analysis.

        Args:
            target (str): Column that will be analysed.
        """

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        newdf = self.data.select_dtypes(include=numerics)

        Y = newdf[target]
        x = newdf[[column for column in newdf.columns if column != target]]

        x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.2)

        # Build the model with the random forest regression algorithm:
        model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
        model.fit(x_train, Y_train)

        shap_values = shap.TreeExplainer(model).shap_values(x_train)
        plt.clf()
        shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)

        plt.clf()
        shap.summary_plot(shap_values, x_train, show=False)
        plt.tight_layout()
        image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
        image_description = 'Shapley Additive Explanations for %s.' % (target)
        plt.savefig('./plots/' + image_name, dpi=400)
        self._doc.add_figure(image_name, image_description)

    def _data_analysis_lime(self, target: str) -> None:
        """Method that applies the Lime analysis.

        Args:
            target (str): Column that will be analysed.
        """

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        newdf = self.data.select_dtypes(include=numerics)
        
        Y = newdf[target]
        x = newdf[[column for column in newdf.columns if column != target]]
        x_featurenames = x.columns

        x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.2)

        # Build the model with the random forest regression algorithm:
        model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
        model.fit(x_train, Y_train)

        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
                    feature_names=x_featurenames,
                    class_names=['quality'],
                    # categorical_features=,
                    # There is no categorical features in this example, otherwise specify them.
                    verbose=True, mode='regression')

        plt.clf()
        exp = explainer.explain_instance(x_test.iloc[0], model.predict)
        exp.as_pyplot_figure()

    def _data_analysis_interpret(self, target: str) -> None:
        """Method that applies the Interpret analysis.

        Args:
            target (str): Column that will be analysed.
        """

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        newdf = self.data.select_dtypes(include=numerics)
        
        Y = newdf[target]
        x = newdf[[column for column in newdf.columns if column != target]]

        x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.2)

        # Build the model with the random forest regression algorithm:
        model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
        model.fit(x_train, Y_train)

        marginal = Marginal().explain_data(x_train, Y_train, name = 'Train Data')
        show(marginal)

    def _data_analysis_corr(self) -> None:
        """Method that applies the Pearson Correlation analysis.
        """

        self._doc.add_subsection('Correlations')

        # Compute the correlation matrix
        corr = self.data.corr().iloc[:, ::-1]
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220., 10.)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.tight_layout()
        image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
        image_description = 'Pearson Correlation Coefficient.'
        plt.savefig('./plots/' + image_name, dpi=400)
        self._doc.add_figure(image_name, image_description)

    def _data_analysis_values(self) -> None:
        """Method that applies a generical value analysis.
        """

        self._doc.add_subsection('Individual Target Analysis')

        for target in self.target:
            self._doc.add_subsubsection(target)

            #Histogram
            image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
            image_description = 'Frequency of %s.' % (target)
            image_data = [self.data[target]]
            image_labels = []
            self._plot_figures([], image_data, self.main_date, 'Frequency', image_name, image_labels, how='hist')
            self._doc.add_figure(image_name, image_description)

            #Golden Features
            df_num_corr = pd.DataFrame(self.data.corr()[target]).drop(target)
            df_num_corr['abs'] = df_num_corr.iloc[:, 0].abs()
            df_num_corr = df_num_corr.sort_values(by=['abs'], ascending=False).head().iloc[:, 0]
            golden_features = pd.DataFrame(df_num_corr)
            golden_features.insert(0, '', golden_features.index)
            golden_features = golden_features.reset_index(drop=True)
            golden_features.columns = ['Columns', 'Correlation']
            table_name = 'Table%s' % (target)
            table_description = 'Golden features for %s.' % (target)
            self._doc.add_table(golden_features, table_name, table_description)

            #Shapley Value
            self._data_analysis_shap(target)

            #Lime
            self._data_analysis_lime(target)

            #Interpret
            self._data_analysis_interpret(target)

    def _format_date(self) -> None:
        """Method that format the `main_date` field.
        """

        self.data[self.main_date] = pd.to_datetime(self.data[self.main_date])

        #Generate a DataFrame with all dates in the original data interval
        list_dates = [min(self.data[self.main_date]) + timedelta(days=x) for
              x in range((max(self.data[self.main_date])-min(self.data[self.main_date])).days + 1)]

        #Create dimensions in self.data_date
        date_field = 'Date'
        self.data_date = pd.DataFrame(list_dates, columns=[date_field])
        self.data_date['year'] = pd.DatetimeIndex(self.data_date[date_field]).year
        self.data_date['quarter'] = pd.DatetimeIndex(self.data_date[date_field]).quarter
        self.data_date['month'] = pd.DatetimeIndex(self.data_date[date_field]).month
        self.data_date['month_name'] = self.data_date['month'] .apply(lambda x: calendar.month_abbr[x])
        self.data_date['week'] = pd.DatetimeIndex(self.data_date[date_field]).week
        self.data_date['day'] = pd.DatetimeIndex(self.data_date[date_field]).day
        self.data_date['year-month'] = self.data_date[date_field].dt.to_period('M').astype('str')
        self.data_date['week_day'] = self.data_date[date_field].dt.dayofweek
        self.data_date['week_day_name'] = self.data_date['week_day'] .apply(lambda x: calendar.day_abbr[x])
        self.data_date['month_name_day'] = pd.to_datetime(self.data_date[date_field]).dt.strftime('%b-%d')
        self.data_date['year_day'] = self.data_date[date_field].dt.dayofyear
        self.data_date[date_field] = self.data_date[date_field].dt.date

        #Create dimensions in self.data
        self.data['year'] = pd.DatetimeIndex(self.data[self.main_date]).year
        self.data['quarter'] = pd.DatetimeIndex(self.data[self.main_date]).quarter
        self.data['month'] = pd.DatetimeIndex(self.data[self.main_date]).month
        self.data['month_name'] = self.data['month'] .apply(lambda x: calendar.month_abbr[x])
        self.data['week'] = pd.DatetimeIndex(self.data[self.main_date]).week
        self.data['day'] = pd.DatetimeIndex(self.data[self.main_date]).day
        self.data['year-month'] = self.data[self.main_date].dt.to_period('M').astype('str')
        self.data['week_day'] = self.data[self.main_date].dt.dayofweek
        self.data['week_day_name'] = self.data['week_day'] .apply(lambda x: calendar.day_abbr[x])
        self.data['month_name_day'] = pd.to_datetime(self.data[self.main_date]).dt.strftime('%b-%d')
        self.data['year_day'] = self.data[self.main_date].dt.dayofyear
        self.data[self.main_date] = self.data[self.main_date].dt.date
        self.data = self.data.sort_values(self.main_date)

    def _dataset_info(self) -> None:
        """Method to get and format Dataframe.info().
        """

        #Table with data info
        self._doc.add_subsection('Dataset Info')
        table_name = 'DatasetInfo'
        table_description = 'Generate info for each column.'
        df_info = pd.DataFrame(columns=['Column', 'Non-Null Count', 'Dtype', 'Nunique'])
        for column in self.data:
            non_null_count = self.data[column].count()
            dtype = self.data[column].dtype
            nunique = self.data[column].nunique()
            df_info.loc[len(df_info)] = [column, non_null_count, dtype, nunique]
        self._doc.add_table(df_info, table_name, table_description)

        table_name = 'DatasetFirstLast'
        table_description = 'First and Last samples.'
        df_info = self.data.head(1).append(self.data.tail(1))
        self._doc.add_table(df_info, table_name, table_description)

    def _dataset_describe(self) -> None:
        """Method to get and format Dataframe.describe().
        """

        #Table with data describe
        self._doc.add_subsection('Dataset Describe')
        table_name = 'DatasetDescribe'
        table_description = 'Describe the columns.'
        self._doc.add_table_index(self.data.describe().round(15), table_name, table_description)

    def _temporal_analysis_calculate(self, function_title: str, agg_function: str, agg_value: str, issub: bool = True):
        """Method that applies Aggregation functions to generate charts about the data distribution over the `main_date`.

        Args:
            function_title (str): Beauty name for the aggregation function to use in the charts titles.
            agg_function (str): Aggregation function to be applied (e.g.: max, sum, mean).
            agg_value (str): Column name for the main Numerical column (Target). `MAY ALTER TO LIST(STR) IN THE FUTURE`
        """

        self._doc.add_subsection(function_title) if issub else None

        if(self.data['year'].nunique() > 1):
            period = self.main_date
            period_text = ''
            image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
            image_description = '%s %s for each %s.' % (period_text, function_title, self.main_date)

            data_grouped_period = self.data.groupby(by=[period]).count().reset_index()[[period]]
            image_data = []
            image_labels = []
            for year in self.data['year'].unique():
                df_filtered = self.data.loc[self.data['year'] == year, :]
                data_grouped = df_filtered.groupby(by=[period]).agg({agg_value:agg_function}).reset_index()

                data_grouped = pd.merge(data_grouped_period, data_grouped, how='left', on=period)

                image_data.append(data_grouped.iloc[:, 1])
                image_labels.append(str(year))

            self._plot_figures(data_grouped_period.iloc[:, 0], image_data,
                              self.main_date, function_title, image_name, image_labels)
            self._doc.add_figure(image_name, image_description)

        for period, period_text in zip(['week', 'month', 'quarter', 'year_day', 'day', 'week_day'], ['Weekly', 'Monthly', 'Quarterly', 'Year Day -', 'Month Day -', 'Week Day -']):
            image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
            image_description = '%s %s for each %s.' % (period_text, function_title, self.main_date)

            data_grouped_period = self.data_date.groupby(by=[period]).count().reset_index()[[period]]
            image_data = []
            image_labels = []
            for year in self.data['year'].unique():
                data_filtered = self.data.loc[self.data['year'] == year, :]
                data_grouped = data_filtered.groupby(by=[period]).agg({agg_value:agg_function}).reset_index()

                data_grouped = pd.merge(data_grouped_period, data_grouped, how='left', on=period)

                image_data.append(data_grouped.iloc[:, 1])
                image_labels.append(str(year))

            if(period == 'week_day'):
                xticks = data_grouped_period.iloc[:, 0].map(self.week_day_name)
            elif(period == 'month'):
                xticks = data_grouped_period.iloc[:, 0].map(self.month_name)
            else:
                xticks =  data_grouped_period.iloc[:, 0]

            how = 'line' if len(image_data[0]) > 100 else 'bar'
            self._plot_figures(xticks, image_data,
                              self.main_date, function_title, image_name, image_labels, how=how)

            self._doc.add_figure(image_name, image_description)

    def _temporal_analysis_consistency(self, function_title: str, agg_function: str, agg_value: str) -> None:
        """Method that applies Aggregation functions to generate charts about the data consistency over the `main_date`.

        Args:
            function_title (str): Beauty name for the aggregation function to use in the charts titles.
            agg_function (str): Aggregation function to be applied (e.g.: max, sum, mean).
            agg_value (str): Column name for the main Numerical column (Target). `MAY ALTER TO LIST(STR) IN THE FUTURE`
        """

        self._doc.add_subsection(function_title)

        #Last 30 Days
        image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
        image_description = 'Count rows for each %s in the last 30 days.' % (self.main_date)
        date_min = max(self.data[self.main_date]) - timedelta(days=30)
        data_date_filtered = self.data_date.loc[self.data_date['Date'] > date_min, :][['Date']]
        data_grouped = self.data.groupby(by=[self.main_date]).agg({agg_value:agg_function}).reset_index()
        data_grouped = pd.merge(data_date_filtered, data_grouped, left_on='Date', right_on=self.main_date, how='left')
        image_data = [data_grouped.iloc[:, 2]]
        image_labels = ['']
        how = 'bar'
        self._plot_figures(data_grouped.iloc[:, 0], image_data,
                          self.main_date, 'Count Samples', image_name, image_labels, how=how)
        self._doc.add_figure(image_name, image_description)

        #Last 12 Months
        image_name = 'figure_' + str(datetime.now()).replace('.', '_').replace(':', '_') + '.png'
        image_description = 'Days without samples in the last 12 months.'
        date_min = max(self.data[self.main_date]) - relativedelta(months=13)

        data_date_filtered = self.data_date.loc[self.data_date['Date'] > date_min, :]
        data_date_grouped = data_date_filtered.groupby(by=['year-month']).agg({'Date':'nunique'}).reset_index()
        data_grouped = self.data.groupby(by=['year-month']).agg({self.main_date:'nunique'}).reset_index()

        data_date_grouped = pd.merge(data_date_grouped, data_grouped, on='year-month', how='left')
        data_date_grouped['__result__'] = data_date_grouped.iloc[:, 1] - data_date_grouped.iloc[:, 2]

        image_data = [data_date_grouped.iloc[:, 3]]
        image_labels = ['']
        how = 'bar'
        self._plot_figures(data_date_grouped.iloc[:, 0], image_data,
                          self.main_date, 'Days Without Samples', image_name, image_labels, how=how)

        self._doc.add_figure(image_name, image_description)

    def _plot_figures(self, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, figure_name: str, image_labels: list, how: str = 'line') -> None:
        """Method that create charts and figures and save them on disk.

        Args:
            x (np.ndarray): The x coordinates.
            y (np.ndarray): The y coordinates.
            xlabel (str): The xlabel text.
            ylabel (str): The ylabel text.
            figure_name (str): The figure title.
            image_labels (list): The image labels.
            how (str, optional): The chart type. Defaults to 'line'.
        """

        bar_width = 0.25 if len(image_labels) == 1 else (1.0 / (len(image_labels) + 1.0))
        r = np.arange(len(y[0]))

        plt.ioff()
        f, ax = plt.subplots()
        plt.clf()
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)

        for i in range(len(y)):
            if(how == 'line'):
                plt.plot(x, y[i], color=self.colors[i], label=image_labels[i])
            elif(how == 'bar'):
                if(len(x) > 12):
                    y = np.array([np.nan_to_num(item) for item in y])
                    bottom = y[:i].sum(axis=0)
                    plt.bar(x, y[i], color=self.colors[i], label=image_labels[i], bottom=bottom)
                else:
                    plt.bar(r, y[i], width=bar_width, color=self.colors[i],
                            label=image_labels[i], align='center', edgecolor='white')
                    r = [x + bar_width for x in r]
            elif(how == 'hist'):
                sns.distplot(y[i], color=self.colors[i], bins=50, hist_kws={'alpha': 0.5, 'edgecolor':self.colors[i], 'linewidth':.5})

        if(how != 'hist'):
            if(how == 'bar' and len(x) <= 12):
                plt.xticks([r + bar_width for r in range(len(y[0]))], x, rotation=90, fontsize=9)
            elif(len(x) > 15):
                indexes = np.array([int(value) for value in np.arange(0, len(x) + 1, len(x) / 15.0)])
                indexes[-1] = len(x) - 1
                plt.xticks(x.iloc[indexes], rotation=90, fontsize=9)
            else:
                plt.xticks(x, rotation=90, fontsize=9)
        else:
            plt.xticks(rotation=90, fontsize=9)

        if(len(y) > 1):
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                       ncol=3, mode="expand", borderaxespad=0.)

        plt.yticks(fontsize=9)
        plt.tight_layout()

        plt.gca().yaxis.grid(True)

        plt.savefig('./plots/' + figure_name, dpi=400)
        plt.close(fig=None)
# %%