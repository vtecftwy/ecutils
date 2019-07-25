import collections
import os
import logging
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib import style
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.io import output_notebook
from scipy import optimize

# Set names for columns headers, independently of possible changes by Mintos.
LOANS_CURRENT_HEADERS = ['Country',
                         'ID',
                         'Issue Date',
                         'Loan Type',
                         'Amortization Method',
                         'Loan Originator',
                         'Loan Amount',
                         'Remaining Principal',
                         'Next Payment',
                         'Estimated Next Payment',
                         'LTV',
                         'Interest Rate',
                         'Remaining Term',
                         'Payments Received',
                         'Status',
                         'Buyback Guarantee',
                         'My Investments',
                         'Date of Investment',
                         'Received Payments',
                         'Outstanding Principal',
                         'Amount in Secondary Market',
                         'Price',
                         'Discount/Premium',
                         'Currency'
                         ]

LOANS_CLOSED_HEADERS = ['ID',
                        'Issue Date',
                        'Loan Type',
                        'Amortization Method',
                        'Loan Originator',
                        'Loan Amount',
                        'Remaining Principal',
                        'LTV',
                        'Interest Rate',
                        'Remaining Term',
                        'Payments Received',
                        'Status',
                        'Buyback Guarantee',
                        'My Investments',
                        'Date of Investment',
                        'Received Payments',
                        'Outstanding Principal',
                        'Amount in Secondary Market',
                        'Price',
                        'Discount/Premium',
                        'Currency',
                        'Finished'
                        ]

STATEMENT_HEADERS = ['Transaction ID',
                     'Date',
                     'Details',
                     'Turnover',
                     'Balance',
                     'Currency'
                     ]


def printlog(text_to_print, print_text=False):
    """Legacy function to replace printlog() into a logging.info()"""
    logging.info(text_to_print)


def initialize_mintos_data(loan_selection=None, verbose=False):
    """
    load Mintos file into DataFrames, cleans up transaction statement and add columns (LoanID and Days)
    no input parameters
    :return: loan_related_transactions, loans_all
    """
    if loan_selection is None:
        loan_selection = 'all'

    # Select latest files from set of files in data folder
    STR_4_STATEMENT = ['account-statement']
    STR_4_CLOSED = ['finished-investments']
    STR_4_CURRENT = ['current-investments']

    data_folder = Path('data')
    inv_folder = data_folder / 'investments'

    data_files = [x for x in inv_folder.iterdir() if x.is_file()]

    statement_file = current_file = closed_file = ''
    last_modif_statement = last_modif_current = last_modif_closed = dt.datetime(year=2000, month=1, day=1)

    for f in data_files:
        file_is_statement = any(string in f.name for string in STR_4_STATEMENT)
        file_is_closed = any(string in f.name for string in STR_4_CLOSED)
        file_is_current = any(string in f.name for string in STR_4_CURRENT)

        if file_is_statement and (dt.datetime.fromtimestamp(os.path.getmtime(f)) > last_modif_statement):
            statement_file = f.name
            last_modif_statement = dt.datetime.fromtimestamp(os.path.getmtime(f))
        if file_is_closed and (dt.datetime.fromtimestamp(os.path.getmtime(f)) > last_modif_closed):
            closed_file = f.name
            last_modif_closed = dt.datetime.fromtimestamp(os.path.getmtime(f))
        if file_is_current and (dt.datetime.fromtimestamp(os.path.getmtime(f)) > last_modif_current):
            current_file = f.name
            last_modif_current = dt.datetime.fromtimestamp(os.path.getmtime(f))

    printlog(f'# Latest data files to be loaded:', print_text=verbose)
    printlog(f'  - {statement_file}  \n  - {current_file}  \n  - {closed_file}', print_text=verbose)

    path_to_statement_file = inv_folder / statement_file
    path_to_closed_file = inv_folder / closed_file
    path_to_current_file = inv_folder / current_file


    # Load data from xlsx files
    printlog('# Load data from xlsx files', verbose)

    mintos_statement = pd.read_excel(path_to_statement_file,
                                     header=0, index_col=0,
                                     parse_dates=[1], dayfirst=True
                                     )
    printlog(' - Loaded account statement', verbose)

    loans_closed = pd.read_excel(path_to_closed_file,
                                 header=0, index_col=1,
                                 parse_dates=[2, 17, 24], dayfirst=True
                                 )
    printlog(' - Loaded finished loans', verbose)

    loans_current = pd.read_excel(path_to_current_file,
                                  header=0, index_col=1,
                                  parse_dates=[2, 18], dayfirst=True
                                  )
    printlog(' - Loaded current loans', verbose)

    # Add length of loan in days to each closed investments. Merge both investment lists
    printlog('# Add loan length in days to each closed investments. Merge both investment lists', verbose)

    loans_closed.loc[:, 'Days'] = (loans_closed.loc[:, 'Finished'] - loans_closed.loc[:, 'Date of Investment']).dt.days
    loans_current.loc[:, 'Days'] = (last_modif_current - loans_current.loc[:, 'Date of Investment']).dt.days

    if loan_selection == 'all':
        loans_all = loans_closed.append(loans_current, sort=True)
    elif loan_selection == 'closed':
        loans_all = loans_closed
    else:
        loans_all = loans_current

    # Create a DF with only loan related transactions (excludes cash in and cash out)
    printlog('# Create a DF with only loan related transactions (excludes cash in and cash out)', verbose)

    loan_related_transactions = mintos_statement[mintos_statement.Details.str.contains('Loan')]

    # Extract Loan ID from each transaction and add an ID column to the loan_related_transactions dataframe
    printlog('# Extract Loan ID from each transaction and add an ID column to the loan_related_transactions dataframe',
             verbose)

    id_capture_group = '(\d{5,15}-\d{1,5})'
    loan_id = loan_related_transactions['Details'].str.extract(id_capture_group, expand=True)
    loan_id.columns = ['ID']
    loan_related_transactions = loan_related_transactions.assign(LoanID=loan_id)

    # Add Loan length in days and Loan equivalent annual growth rate to loans_all DF
    printlog('# Add Loan length in days and Loan equivalent annual growth rate to loans_all DF', verbose)

    lrt = loan_related_transactions
    for loan_id in loans_all.index:
        principal = loans_all.loc[loan_id, 'My Investments']
        pnl = lrt[lrt.LoanID.str.contains(str(loan_id))].Turnover.sum()
        loans_all.loc[loan_id, 'PnL'] = pnl
        days = loans_all.loc[loan_id, 'Days']
        if np.isnan(days):
            rate = 0
        else:
            days = days if days != 0 else 1
            rate = ((principal + pnl) / principal) ** (365 / days) - 1

        loans_all.loc[loan_id, 'Rate'] = rate

    printlog('# All data loaded and processed.', verbose)

    return loan_related_transactions, loans_all


def print_summary_stats(transactions, loans_all, verbose=False):
    """
    Print summary statistics for all transactions and loans provided as parameter
    :param transactions: DF with all transactions to be handled by function 
    :param loans_all: DF with list of all loans to be handled by function
    :param verbose: flag to allow log printing in troubleshoot mode
    :return: none
    """
    # Define Patterns to find for each type of transaction
    printlog('# Define Patterns to find for each type of transaction', verbose)

    INVEST_PRINCIPAL = 'Investment principal increase'
    REPAY_PRINCIPAL = 'Investment principal repayment'
    BUYBACK_PRINCIPAL = 'Investment principal rebuy'
    INTEREST = 'Interest income Loan'
    LATE_INTEREST = 'Delayed interest income Loan'
    INTEREST_BUYBACK = 'Interest income on rebuy'
    LATE_INTEREST_BUYBACK = 'Delayed interest income on rebuy'

    printlog('# Compute Totals', verbose)
    loans_opened = loans_all[loans_all['Outstanding Principal'] > 0]

    tot_capital_paid_in = transactions[transactions.Details.str.startswith(INVEST_PRINCIPAL)].Turnover.sum()
    tot_capital_reimbursed = transactions[transactions.Details.str.startswith(REPAY_PRINCIPAL)].Turnover.sum()
    tot_capital_buyback = transactions[transactions.Details.str.startswith(BUYBACK_PRINCIPAL)].Turnover.sum()
    tot_capital_currently_invested = loans_opened['Outstanding Principal'].sum()
    tot_interest_collected = transactions[transactions.Details.str.startswith(INTEREST)].Turnover.sum()
    tot_interest_late = transactions[transactions.Details.str.startswith(LATE_INTEREST)].Turnover.sum()
    tot_interest_on_buyback = transactions[transactions.Details.str.startswith(INTEREST_BUYBACK)].Turnover.sum()
    tot_interest_late_buyback = transactions[transactions.Details.str.startswith(LATE_INTEREST_BUYBACK)].Turnover.sum()
    tot_interest = tot_interest_collected + tot_interest_late + tot_interest_on_buyback + tot_interest_late_buyback

    print('\n Total Capital Paid In :      {0:9,.1f}'
          '\n Total Capital Reimbursed:    {1:9,.1f}'
          '\n Total Capital Bought Back:   {2:9,.1f}'
          '\n Total Currently Invested:    {3:9,.1f}'
          '\n Balance Capital: Gain/(Loss):{4:9,.1f}'.format(tot_capital_paid_in,
                                                             tot_capital_reimbursed,
                                                             tot_capital_buyback,
                                                             tot_capital_currently_invested,
                                                             tot_capital_paid_in +
                                                             tot_capital_reimbursed +
                                                             tot_capital_buyback +
                                                             tot_capital_currently_invested)
          )

    print('\n Total Interest Collected: {0:6,.1f}'
          '\n Total Late Interest:      {1:6,.1f}'
          '\n Total Interest on Buyback:{2:6,.1f}'
          '\n Total Late on Buyback:    {3:6,.1f}'
          '\n Total Interest:           {4:6,.1f}'
          '\n Balance:                  {5:6,.1f}'.format(tot_interest_collected,
                                                          tot_interest_late,
                                                          tot_interest_on_buyback,
                                                          tot_interest_late_buyback,
                                                          tot_interest,
                                                          tot_interest -
                                                          tot_interest_collected -
                                                          tot_interest_late -
                                                          tot_interest_on_buyback -
                                                          tot_interest_late_buyback
                                                          )
          )

    loans_closed = loans_all[loans_all['Outstanding Principal'] < 1e-8]
    low_yield_loans = loans_closed.loc[loans_closed.loc[:, 'Rate'] < 9e-2]
    normal_yield_loans = loans_closed.loc[loans_closed.loc[:, 'Rate'] >= 9e-2]
    loan_premature = loans_all[loans_all.Status.str.contains('prematurely')]

    loans_per_country = loans_closed['Country'].value_counts()
    low_yield_loans_per_country = low_yield_loans['Country'].value_counts()
    low_yield_loans_percent_per_country = low_yield_loans_per_country / loans_per_country
    normal_yield_loans_per_country = normal_yield_loans['Country'].value_counts()
    normal_yield_loans_percent_per_country = normal_yield_loans_per_country / loans_per_country
    loans_premature_per_country = loan_premature['Country'].value_counts()
    loans_premature_percent_per_country = loans_premature_per_country / loans_per_country
    loans_stats_per_country = pd.DataFrame({'All Loans': loans_per_country,
                                            'Normal Yield': normal_yield_loans_per_country,
                                            'Normal Yield %': normal_yield_loans_percent_per_country,
                                            'Low Yield': low_yield_loans_per_country,
                                            'Low Yield %': low_yield_loans_percent_per_country,
                                            'Premature': loans_premature_per_country,
                                            'Premature %': loans_premature_percent_per_country
                                            })

    print(loans_stats_per_country)


def print_transactions_per_loan(transactions, loans_all, verbose=False):
    """
    Print all all transactions grouped by loan
    :param transactions: DF with all transactions to be considered
    :param loans_all: DF with list of all loans
    :param verbose: flag to allow log printing in troubleshoot mode
    :return: none
    """
    loans_closed = loans_all[loans_all['Outstanding Principal'] < 1e-8]

    for loan_id in loans_closed.index:
        print(loan_id)
        transactions_to_print = transactions[transactions.LoanID.str.contains(str(loan_id))][['Date', 'Turnover']]
        print(transactions_to_print)
        print('Days:', loans_all.loc[loan_id, 'Days'], 'Rate:', loans_all.loc[loan_id, 'Rate'] * 100,
              'Status:', loans_all.loc[loan_id,'Status'])
        print('--------------------')


def get_transactions_per_loan(transactions, loans_all, verbose=False):
    """
    Creates and returns an ordered dictionary with all the transaction grouped by loan
    :param transactions: DF with all transactions to be considered
    :param loans_all: DF with list of all loans
    :param verbose: flag to allow log printing in troubleshoot mode
    :return: transactions_per_loan: the dictionary
    """
    transactions_per_loan = collections.OrderedDict()
    loans_closed = loans_all[loans_all['Outstanding Principal'] < 1e-8]

    for loan_id in loans_closed.index:
        transaction_set = transactions[transactions.LoanID.str.contains(str(loan_id))]
        transactions_per_loan[str(loan_id)] = transaction_set

    return transactions_per_loan


def get_loans_per_country(loans_all, verbose=False):
    """
    Creates and returns a dictionary with all the loans grouped by country
    :param loans_all: DF with list of all loans
    :param verbose: flag to allow log printing in troubleshoot mode
    :return: loans_per_country: the dictionary
    """
    printlog('# Getting loans organized per country')
    loans_closed = loans_all[loans_all['Outstanding Principal'] < 1e-8]
    country_list = loans_closed['Country'].unique()
    loans_per_country = {}

    for cntry in country_list:
        loans_set = loans_closed[loans_closed['Country'] == cntry]
        loans_per_country[str(cntry)] = loans_set
    printlog('# Done organizing loans per country')
    return loans_per_country


def plot_transactions_mpl(transactions, loans_all, fig=None, verbose = False):

    # Set function parameters
    plot_set_count = 0  # different from 0 when more than one subplots in a figure
    style.use(str('ggplot'))
    figsize = (12, 6)
    grid_rows = 3
    grid_columns = 2
    is_hist_cumulative = False
    hist_style = 'step'  # 'step' 'bar'
    font_titles = 8
    font_axes = 8
    adjust_wspace = 0.3
    adjust_hspace = 0.5

    if fig is None:
        # When no figure was opened and passed to the function yet
        # grid_rows = 1
        # grid_columns = 3
        fig = plt.figure(num=0 ,figsize=figsize)
        show_plot_within = True
    else:
        show_plot_within = False

    # Utility function to set subplots
    def set_hist_plot_ax(data_to_plot,
                         count,
                         plot_title='Title',
                         x_label='x',
                         y_label='y',
                         figure=fig,
                         row=grid_rows,
                         col=grid_columns,
                         h_style=hist_style,
                         h_bins=20,
                         is_cum=is_hist_cumulative,
                         ft_titles=font_titles,
                         ft_axes=font_axes):

        ax = figure.add_subplot(row, col, count)
        ax.hist(data_to_plot,
                bins=h_bins,
                histtype=h_style,
                cumulative=is_cum,
                orientation='vertical',
                # color='g',
                alpha=0.75)
        ax.set_title(plot_title, fontsize=ft_titles)
        ax.set_xlabel(x_label, fontsize=ft_axes)
        ax.set_ylabel(y_label, fontsize=ft_axes)
        for label in ax.xaxis.get_ticklabels():
            label.set_fontsize(ft_axes)
        for label in ax.yaxis.get_ticklabels():
            label.set_fontsize(ft_axes)

    def set_bar_plot_ax(data_to_plot,
                        count,
                        plot_title='Title',
                        x_label='x',
                        y_label='y',
                        figure=fig,
                        row=grid_rows,
                        col=grid_columns,
                        ft_titles=font_titles,
                        ft_axes=font_axes):

        x = np.arange(len(data_to_plot))
        y = data_to_plot.values
        bar_labels = data_to_plot.index
        ax = figure.add_subplot(row, col, count)
        ax.barh(x,
                y,
                tick_label=bar_labels,
                # color='g',
                alpha=0.75)
        ax.set_title(plot_title, fontsize=ft_titles)
        ax.set_xlabel(x_label, fontsize=ft_axes)
        ax.set_ylabel(y_label, fontsize=ft_axes)
        for label in ax.xaxis.get_ticklabels():
            label.set_fontsize(ft_axes)
        for label in ax.yaxis.get_ticklabels():
            label.set_fontsize(ft_axes)

    max_grid_position = grid_rows * grid_columns
    requested_first_position = plot_set_count * grid_columns + 1
    if requested_first_position > max_grid_position:
        print('---------------------------------------------')
        print('| Figure is full, cannot print more subplot |')
        print('---------------------------------------------')
        return

    # Prepare data to plot
    loans_closed = loans_all[loans_all['Outstanding Principal'] < 1e-8]
    low_yield_loans = loans_closed.loc[loans_closed.loc[:, 'Rate'] < 8e-2]
    normal_yield_loans = loans_closed.loc[loans_closed.loc[:, 'Rate'] >= 8e-2]

    loans_per_country = loans_closed['Country'].value_counts()
    low_yield_loan_per_country = low_yield_loans['Country'].value_counts()
    normal_yield_loan_per_country = normal_yield_loans['Country'].value_counts()

    data_to_plot_low_yield_days = low_yield_loans['Days']
    data_to_plot_normal_yield_days = normal_yield_loans['Days']

    data_to_plot_low_yield_country = low_yield_loan_per_country
    data_to_plot_normal_yield_country = normal_yield_loan_per_country

    # <> To Do: Correct these four lines
    # low_yield_ratio = low_yield_loan_per_country / loans_per_country if loans_per_country != 0 else 0
    # normal_yield_ratio = normal_yield_loan_per_country / loans_per_country if loans_per_country != 0 else 0
    # data_to_plot_low_yield_ratio = low_yield_ratio[low_yield_ratio is not 'NaN']
    # data_to_plot_normal_yield_ratio = normal_yield_ratio[normal_yield_ratio is not 'NaN']

    set_hist_plot_ax(data_to_plot=data_to_plot_low_yield_days,
                     count=0 * grid_columns + 1,
                     plot_title='Length Low Yield Loans',
                     x_label='Days',
                     y_label='Nbr Loans')

    set_hist_plot_ax(data_to_plot=data_to_plot_normal_yield_days,
                     count=0 * grid_columns + 2,
                     plot_title='Length Normal Yield Loans',
                     x_label='Days',
                     y_label='Nbr Loans')

    set_bar_plot_ax(data_to_plot=data_to_plot_low_yield_country,
                    count=1 * grid_columns + 1,
                    plot_title='Country Low Yield Loans',
                    x_label='Nbr Loans',
                    y_label='Countries')

    set_bar_plot_ax(data_to_plot=data_to_plot_normal_yield_country,
                    count=1 * grid_columns + 2,
                    plot_title='Country Normal  Yield Loans',
                    x_label='Nbr Loans',
                    y_label='Countries')

    # plt.tight_layout()
    if show_plot_within is True:
        plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.90,
                            wspace=0.3, hspace=0.5)
        plt.show()
    else:
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=adjust_wspace, hspace=adjust_hspace)


def plot_transactions_bokeh(transactions, loans_all, fig=None, verbose = False):
    """

    :param transactions:
    :param loans_all:
    :param fig:
    :param verbose:
    :return:
    """
    mask = loans_all.loc[:, 'Status'].str.contains('Finished')
    loan_subset = loans_all.loc[mask, ['Country', 'Days', 'My Investments', 'Received Payments', 'PnL', 'Rate', 'Status']]

    lbc = loan_subset.groupby(['Country'])
    lbcs = loan_subset.groupby(['Country', 'Status'])

    size = 10
    alpha = 0.3

    source_data_2 = ColumnDataSource(data=lbc)
    index_list_2 = list(source_data_2.data['Country'])
    p2 = figure(title='Per Country: All Loans',
                x_range=index_list_2,
                y_range=(0, 0.14),
                plot_width=800,
                plot_height=400,
                )
    p2.vbar(x='Country', bottom='Rate_25%', top='Rate_75%',
            color='navy', width=.3, alpha=alpha, source=source_data_2)
    p2.circle(x='Country', y='Rate_mean', size=size, color='navy', alpha=alpha*2, source=source_data_2)
    p2.xaxis.major_label_orientation = 3.1415 / 4
    p2.yaxis.axis_label = "Return Rates"

    source_data_3 = ColumnDataSource(data=lbcs)
    index_list = list(source_data_3.data['Country_Status'])
    p3 = figure(title='Per Country: Matured Loans and Premature',
                x_range=FactorRange(*index_list),
                y_range=(0, 0.14),
                plot_width=800,
                plot_height=600,
                )
    p3.vbar(x='Country_Status', bottom='Rate_25%', top='Rate_75%',
            color='navy', width=.5, alpha=alpha, source=source_data_3,
            legend='25%-75% Range')
    p3.circle(x='Country_Status', y='Rate_mean',
              size=size, color='navy', alpha=alpha * 2, source=source_data_3,
              legend='Mean')

    p3.xaxis.major_label_orientation = 3.1415 / 2
    p3.xaxis.group_label_orientation = 3.1315 / 2
    p3.yaxis.axis_label = "Return Rates"
    p3.legend.location = 'bottom_center'
    p3.legend.orientation = "horizontal"

    show(p2)
    show(p3)


# FINANCIAL FUNCTIONS
# 1. Extended IRR functions. From GitHub: https://github.com/peliot/XIRR-and-XNPV
def secant_method(tol, f, x0):
    """
    Solve for x where f(x)=0, given starting x0 and tolerance.

    Arguments
    ----------
    tol: tolerance as percentage of final result. If two subsequent x values are with tol percent, the function will return.
    f: a function of a single variable
    x0: a starting value of x to begin the solver

    Notes
    ------
    The secant method for finding the zero value of a function uses the following formula to find subsequent values of x. 

    x(n+1) = x(n) - f(x(n))*(x(n)-x(n-1))/(f(x(n))-f(x(n-1)))

    Warning 
    --------
    This implementation is simple and does not handle cases where there is no solution. Users requiring a more robust version should use scipy package optimize.newton.

    """
    x1 = x0 * 1.1
    while abs(x1 - x0) / abs(x1) > tol:
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    return x1


def xnpv(rate, cashflows):
    """
    Calculate the net present value of a series of cashflows at irregular intervals.

    Arguments
    ---------
     rate: the discount rate to be applied to the cash flows
     cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.

    Returns
    -------
     returns a single value which is the NPV of the given cash flows.

    Notes
    ---------------
     The Net Present Value is the sum of each of cash flows discounted back to the date of the first cash flow. The discounted value of a given cash flow is A/(1+r)**(t-t0), where A is the amount, r is the discout rate, and (t-t0) is the time in years from the date of the first cash flow in the series (t0) to the date of the cash flow being added to the sum (t).
     This function is equivalent to the Microsoft Excel function of the same name.

    """

    chron_order = sorted(cashflows, key=lambda x: x[0])
    t0 = chron_order[0][0]  # t0 is the date of the first cash flow

    return sum([cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in chron_order])


def xirr(cashflows, guess=0.1):
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.

    Arguments
    ---------
     :param cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python
     datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are
     represented with negative amounts, and cash inflows (returns) are positive amounts.
     :param guess: (optional, default = 0.1): a guess at the solution to be used as a starting point for the numerical solution.

    Returns
    --------
     Returns the IRR as a single value

    Notes
    ----------------
     The Internal Rate of Return (IRR) is the discount rate at which the Net Present Value (NPV) of a series of cash
     flows is equal to zero. The NPV of the series of cash flows is determined using the xnpv function in this module.
     The discount rate at which NPV equals zero is found using the secant method of numerical solution.
     This function is equivalent to the Microsoft Excel function of the same name.
     For users that do not have the scipy module installed, there is an alternate version (commented out) that uses
     the secant_method function defined in the module rather than the scipy.optimize module's numerical solver.
     Both use the same method of calculation so there should be no difference in performance, but the secant_method
     function does not fail gracefully in cases where there is no solution, so the scipy.optimize.newton version
     is preferred.

    """

    # return secant_method(0.0001,lambda r: xnpv(r,cashflows),guess)
    return optimize.newton(lambda r: xnpv(r, cashflows), guess)


def loan_irr(loans_to_analyse):
    """Wrapper around xirr to prepare cashflows tuple list from list of loans"""

    outflows = pd.DataFrame(loans_to_analyse.loc[:,['Date of Investment', 'My Investments']].values, columns=['Date', 'Cashflows'])
    outflows.loc[:,'Cashflows'] = (- 1) * outflows.loc[:,'Cashflows']
    inflows = pd.DataFrame(loans_to_analyse.loc[:,['Finished', 'Received Payments']].values, columns=['Date', 'Cashflows'])
    cashflows = outflows.append(inflows, sort=True)
    cashflows_list = [(df.loc['Date'], df.loc['Cashflows']) for __, df in cashflows.iterrows()]
    ext_irr = xirr(cashflows_list)
    return ext_irr

# END OF FINANCIAL FUNCTIONS


if __name__ == '__main__':

    loan_transactions, loans = initialize_mintos_data(loan_selection='current', verbose=True)

    # sets_per_country = get_loans_per_country(loans, verbose=True)
    # for country, loan_set in sets_per_country.items():
    #     irr = loan_irr(loan_set)
    #     print('{description:.<20s}{value:>7.2%}'.format(description=country, value=irr))
    #
    # loans_all = loans
    #
    # yield_treshold = 8e-2
    # loans_closed = loans_all[loans_all['Outstanding Principal'] < 1e-8]
    # low_yield_loans = loans_closed.loc[loans_closed.loc[:, 'Rate'] < yield_treshold]
    # normal_yield_loans = loans_closed.loc[loans_closed.loc[:, 'Rate'] >= yield_treshold]
    # loan_premature = loans_all[loans_all.Status.str.contains('prematurely')]
    # loans_per_country = get_loans_per_country(loans_all)
    #
    # loans_sets_1 = {'Low yield': low_yield_loans, 'Normal yield': normal_yield_loans, 'Premature': loan_premature}
    #
    # print('xxx')
    # for text, loan_set in loans_sets_1.items():
    #     irr = loan_irr(loan_set)
    #     print('{description:.<20s}{value:>7.2%}'.format(description=text, value=irr))
    #
    # # print('SUMMARY STATS AND TOTALS')
    # # print_summary_stats(transactions=loan_transactions, loans_all=loans)
    # #
    # # # print('LIST OF LOANS WITH DETAILS')
    # # # print_transactions_per_loan(transactions=loan_transactions, loans_all=loans)
    # #
    # # figsize = (12, 6)
    # # fig1 = plt.figure(num=0, figsize=figsize)
    # # plot_transactions(transactions=loan_transactions,loans_all=loans, fig=fig1)
    # # plt.show()
    #
    # plot_transactions_bokeh(loan_transactions, loans)