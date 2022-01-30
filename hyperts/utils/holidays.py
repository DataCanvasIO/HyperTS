# import chinese_calendar

def get_holidays(year=None, include_weekends=True):
#     """
#     Parameters
#     ----------
#     year: which year
#     include_weekends: False for excluding Saturdays and Sundays
#
#     Returns
#     -------
#     A list.
#     """
#     if not year:
#         year = datetime.datetime.now().year
#     else:
#         year = year
#     start = datetime.date(year, 1, 1)
#     end = datetime.date(year, 12, 31)
#     holidays = chinese_calendar.get_holidays(start, end, include_weekends)
#     holidays = pd.DataFrame(holidays, columns=['Date'])
#     holidays['Date'] = holidays['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
#     return holidays
    pass