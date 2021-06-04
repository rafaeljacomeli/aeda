__doc__ = ''
__version__ = '0.0.1'

import os
import subprocess
import pandas as pd

path = os.getcwd()

class TexBuilder():
    """TexBuilder is a Python library to generate and build a PDF and a LaTeX file.
    """

    def __init__(self, report_name: str, report_title: str) -> None:
        """Constructor method.

        Args:
            report_name (str): Report name.
            report_title (str): Report title.
        """

        self.report_name = report_name
        self.report_title = report_title
        self.page_info = r'by Jacob'

        self.table_columns_count = 4

        self.tables = {}
        self.figures = {}

        self.doc = ''

    def add_space(self) -> None:
        """Add `space` tag in LaTeX file.
        """

        tag = r'\vspace{5mm}'
        self.doc = self.doc + tag

    def add_chapter(self, text: str) -> None:
        """Add `chapter` tag in LaTeX file.

        Args:
            text (str): Chapter text.
        """

        tag = r'\chapter{%s}' % (text)
        self.doc = self.doc + tag

    def add_section(self, text: str) -> None:
        """Add `section` tag in LaTeX file.

        Args:
            text (str): Section text.
        """

        tag = r'''\newpage
        \section{%s}''' % (text)
        self.doc = self.doc + tag

    def add_subsection(self, text: str) -> None:
        """Add `subsection` tag in LaTeX file.

        Args:
            text (str): Subsection text.
        """

        tag = r'\subsection{%s}' % (text)
        self.doc = self.doc + tag

    def add_subsubsection(self, text: str) -> None:
        """Add `subsubsection` tag in LaTeX file.

        Args:
            text (str): Subsubsection text.
        """

        tag = r'\subsubsection{%s}' % (text)
        self.doc = self.doc + tag

    def add_paragraph(self, text: str) -> None:
        """Add `par` tag in LaTeX file.

        Args:
            text (str): Paragraph text.
        """

        tag = r'\par ' + text
        self.doc = self.doc + tag
        # self.add_space()

    def add_figure(self, figure_name: str, figure_caption: str) -> None:
        """Add edited `figure` tag in LaTeX file.

        Args:
            figure_name (str): Figure name.
            figure_caption (str): Figure caption.
        """

        tag = r'''
        \begin{figure}[H]
            \centering
            \caption{''' + figure_caption + '''}
            \label{fig:''' + figure_name + '''}
            \includegraphics[width=400px]{../plots/''' + figure_name + '''}
        \end{figure}
        '''

        self.figures[len(self.figures)] = [figure_name, figure_caption]

        self.doc = self.doc + tag

    def add_table(self, table: pd.DataFrame, table_name: str, table_description: str) -> None:
        """Add edited `table` or `longtable` tag in LaTeX file.

        Args:
            table (pd.DataFrame): Dataset to convert as a table.
            table_name (str): Table name.
            table_description (str): Table description.
        """

        columns = table.columns

        columns_lists = []
        for i in range(0, len(columns), self.table_columns_count):
            columns_list = []
            for j in range(i, len(columns)):
                columns_list.append(columns[j])
                if(len(columns_list) == self.table_columns_count):
                    break
            columns_lists.append(columns_list)

        tag = r""
        for i in range(len(columns_lists)):
            columns_list = columns_lists[i]

            if(len(columns_lists) == 1):
                table_num = r''
            else:
                table_num = r' [' + str(i + 1) + r'/' + str(len(columns_lists)) + r']'

            if(len(table) < 30):
                tag = tag + r'''
                \begin{table}[H]
                \center
                \caption{''' + table_description + table_num + '''}
                \label{tab:''' + table_name + str(len(self.tables)) + r'''}
                \begin{tabular}{c''' + ' c' * len(columns_list) + '''}
                '''
            else:
                tag = tag + r'''
                \begin{longtable}{''' + 'c ' + ' c' * len(columns_list) + '''}
                \caption{''' + table_description + table_num + '''\label{tab:''' + table_name + str(len(self.tables)) + r'''}}\\
                '''

            cell = str(columns_list[0])

            for column in columns_list[1:]:
                cell = cell + r' & ' + str(column)
            tag = tag + cell + r''' \\

            \hline
            '''

            for j in range(len(table)):
                cell = str(table[columns_list[0]].values[j])

                for column in columns_list[1:]:
                    cell = cell + r' & ' + str(table[column].values[j])

                tag = tag + cell + r''' \\
                '''

            if(len(table) < 30):
                tag = tag + r'''
                \hline
                \end{tabular}
                \end{table}
                '''
            else:
                tag = tag + r'''
                \hline
                \end{longtable}
                '''

            self.tables[len(self.tables)] = [table_name, table_description]

        tag = tag.replace('%', '\%').replace('_', '\_').replace('#', '\#')

        self.doc = self.doc + tag

    def add_table_index(self, table: pd.DataFrame, table_name: str, table_description: str) -> None:
        """Add edited indexed `table` or `longtable` tag in LaTeX file.

        Args:
            table (pd.DataFrame): Dataset to convert as a table.
            table_name (str): Table name.
            table_description (str): Table description.
        """

        indexes = table.index
        columns = table.columns

        columns_lists = []
        for i in range(0, len(columns), self.table_columns_count):
            columns_list = []
            for j in range(i, len(columns)):
                columns_list.append(columns[j])
                if(len(columns_list) == self.table_columns_count):
                    break
            columns_lists.append(columns_list)

        tag = r''
        for i in range(len(columns_lists)):
            columns_list = columns_lists[i]

            if(len(columns_lists) == 1):
                table_num = r''
            else:
                table_num = r' [' + str(i + 1) + '/' + str(len(columns_lists)) + ']'

            if(len(table) < 30):
                tag = tag + r'''
                \begin{table}[H]
                \center
                \caption{''' + table_description + table_num + '''}
                \label{tab:''' + table_name + str(len(self.tables)) + r'''}
                \begin{tabular}{c''' + ' c' * len(columns_list) + '''}
                '''
            else:
                tag = tag + r'''
                \begin{longtable}{''' + 'c ' + ' c' * len(columns_list) + '''}
                \caption{''' + table_description + table_num + '''\label{tab:''' + table_name + str(len(self.tables)) + r'''}}\\
                '''

            cell = r'''
                & ''' + str(columns_list[0])

            for column in columns_list[1:]:
                cell = cell + r' & ' + str(column)
            tag = tag + cell + r''' \\

            \hline
            '''

            for index in indexes:
                cell = str(index)

                for column in columns_list:
                    cell = cell + r' & ' + str(table.loc[index, column])

                tag = tag + cell + r''' \\
                '''

            if(len(table) < 30):
                tag = tag + r'''
                \hline
                \end{tabular}
                \end{table}
                '''
            else:
                tag = tag + r'''
                \hline
                \end{longtable}
                '''

            self.tables[len(self.tables)] = [table_name, table_description]

        tag = tag.replace('%', '\%').replace('_', '\_').replace('#', '\#')

        self.doc = self.doc + tag

    def start(self) -> None:
        """Method to initialize the LaTeX file.
        """

        self.doc = self.doc + r'''
        \documentclass[
        10pt, % Main document font size
        a4paper, % Paper type, use 'letterpaper' for US Letter paper
        ]{scrartcl}

        \usepackage{graphicx}
        \usepackage{epstopdf}
        \usepackage{float}
        \usepackage[scale=0.75]{geometry} % Reduce document margins
        \usepackage{hyperref}
        \usepackage{longtable}

        \begin{document}

        \title{Automatic Exploratory Data Analysis} % The article title

        \subtitle{Study Case} % Uncomment to display a subtitle

        \author{Jacob} % The article author(s) - author affiliations need to be specified in the AUTHOR AFFILIATIONS block\

        \maketitle % Print the title/author/date block

        \newpage
        \tableofcontents % Print the table of contents

        \newpage
        \listoffigures % Print the list of figures

        \newpage
        \listoftables % Print the list of tables
        '''

    def build(self) -> None:
        """Method that build the LaTeX and PDF files.

        Raises:
            ValueError: Error when compiling the file.
        """

        print("Genereting files..")
        self.doc = self.doc + r'\end{document}'

        f = open("latex\\" + self.report_name + '.tex', 'w')
        f.write(self.doc)
        f.close()

        os.chdir('latex')

        cmd = ['pdflatex', '-interaction', 'nonstopmode', self.report_name + '.tex']
        #cmd = ['pdflatex', '-interaction', self.report_name + '.tex']

        for i in range(2):
            proc = subprocess.Popen(cmd)
            proc.communicate()
            retcode = proc.returncode
            if not retcode == 0:
                os.chdir('..')
                raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

        os.unlink(self.report_name + '.aux')
        os.unlink(self.report_name + '.lof')
        os.unlink(self.report_name + '.log')
        os.unlink(self.report_name + '.lot')
        os.unlink(self.report_name + '.out')
        os.unlink(self.report_name + '.toc')

        os.chdir('..')