from pathlib import Path

""" Configuration dictionaries related to historical prices raw storage and datasets """
NAS_raw_data = Path('R:\\Financial Data\\Historical Prices Raw Data\\alphavantage')
NAS_datasets = Path('R:\\Financial Data\\Historical Prices Datasets\\alphavantage')

sources_dict = {'alphavantage': {'name': 'alphavantage',
                                 'directory': 'alphavantage',
                                 'format': 'alphavantage'},
                'axitrader': {'name': 'axitrader',
                              'directory': 'axitrader-mt4',
                              'format': 'mt4'},
                'yahoo': {'name': 'yahoo',
                          'directory': 'yahoo',
                          'format': 'yahoo'},
                'xm-com': {'name': 'xm-com',
                           'directory': 'xm-com-mt4',
                           'format': 'mt4'},
                'metatrader': {'name': 'metatrader',
                               'directory': 'metatrader-mt4',
                               'format': 'mt4'},
                'mixed-mt4': {'name': 'mixed-mt4',
                              'directory': 'mixed-mt4',
                              'format': 'mt4'},
                'wsj': {'name': 'wsj',
                        'directory': 'wsj',
                        'format': 'wsj'},
                }

ticker_dict = {'S&P500': {'name': 'S&P500',
                      'type': 'index',
                      'description': 'Standard and Poor 500',
                      'axitrader': 'US500',
                      'wsj': 'SPX',
                      'yahoo': '^GSPC',
                      },
               'S&P500 option': {'name': 'S&P500 option',
                                 'type': 'option',
                                 'description': 'Standard and Poor 500 option',
                                 'axitrader': 'S&P.fs',
                                 'wsj': '-',
                                 'yahoo': '-',
                                 },
               'DJ30': {'name': 'Dow Jones 30',
                        'type': 'index',
                        'description': 'Nyse Composite',
                        'axitrader': 'US30',
                        'wsj': 'NYA',
                        'yahoo': '^NYA',
                        },
               'DJ30 option': {'name': 'Dow Jones 30 option',
                               'type': 'option',
                               'description': 'Nyse Composite option',
                               'axitrader': 'DJ30.fs',
                               'wsj': '-',
                               'yahoo': '-',
                               },
               'NASDAQ100': {'name': 'NASDAQ 100',
                             'type': 'index',
                             'description': 'NASDAQ 100',
                             'axitrader': 'USTECH',
                             'wsj': 'NDX',
                             'yahoo': '^NDX',
                             },
               'NASDAQ100 option': {'name': 'NASDAQ 100 option',
                                    'type': 'option',
                                    'description': 'NASDAQ 100',
                                    'axitrader': 'NAS100.fs',
                                    'wsj': '-',
                                    'yahoo': '-',
                                    },
               'EUSTOXX50': {'name': 'Euro Stoxx 50',
                             'type': 'index',
                             'description': 'Euro Stoxx 50 components',
                             'axitrader': 'EU50',
                             'wsj': 'SX5E',
                             'yahoo': '^STOXX50E',
                             },
               'EUSTOXX50 option': {'name': 'Euro Stoxx 50 option',
                                    'type': 'option',
                                    'description': 'Euro Stoxx 50 components option',
                                    'axitrader': 'EUSTX50.fs',
                                    'wsj': '-',
                                    'yahoo': '-',
                                    },
               'FTSE100': {'name': 'FTSE 100',
                           'type': 'index',
                           'description': 'FTSE Index 100',
                           'axitrader': 'UK100',
                           'wsj': 'UKX',
                           'yahoo': '^FTSE',
                           },
               'FTSE100 option': {'name': 'FTSE 100 option',
                                  'type': 'option',
                                  'description': 'FTSE Index 100 option',
                                  'axitrader': 'FT100.fs',
                                  'wsj': '-',
                                  'yahoo': '-',
                                  },
               'Russel2000': {'name': 'Russel 2000',
                              'type': 'index',
                              'description': 'Russel 2000 index',
                              'axitrader': 'US2000',
                              'wsj': 'RUT',
                              'yahoo': '^RUT',
                              },
               'DAX30': {'name': 'DAX 30',
                         'type': 'index',
                         'description': 'Germany DAX 30 index',
                         'axitrader': 'GER30',
                         'wsj': 'DAX',
                         'yahoo': '^GDAXI',
                         },
               'DAX30 option': {'name': 'DAX 30 option',
                                'type': 'option',
                                'description': 'Germany DAX 30 index option',
                                'axitrader': 'DAX30.fs',
                                'wsj': '-',
                                'yahoo': '-',
                                },
               'CAC40': {'name': 'CAC 40',
                         'type': 'index',
                         'description': 'France CAC 40',
                         'axitrader': 'FRA40',
                         'wsj': 'PX1',
                         'yahoo': '^FCHI',
                         },
               'CAC40 option': {'name': 'CAC 40',
                                'type': 'option',
                                'description': 'France CAC 40 option',
                                'axitrader': 'CAC40.fs',
                                'wsj': '-',
                                'yahoo': '-',
                                },
               'CN50': {'name': 'China 50',
                         'type': 'index',
                         'description': 'China 50',
                         'axitrader': 'CN50',
                         'wsj': '-',
                         'yahoo': '-',
                         },
               'CN50 option': {'name': 'China 50 option',
                               'type': 'option',
                               'description': 'China 50 option',
                               'axitrader': 'CHINA50.fs',
                               'wsj': '-',
                               'yahoo': '-',
                               },
               'Gold USD': {'name': 'Gold (USD)',
                            'type': 'commodity',
                            'description': 'Gold in USD',
                            'axitrader': 'XAUUSD',
                            'wsj': '-',
                            'yahoo': '-',
                            },
               'Silver USD': {'name': 'Silver (USD)',
                              'type': 'commodity',
                              'description': 'Silver in USD',
                              'axitrader': 'XAGUSD',
                              'wsj': '-',
                              'yahoo': '-',
                              },
               'UK Oil': {'name': 'Brent (USD)',
                          'type': 'commodity',
                          'description': 'Brent Crude Oil',
                          'axitrader': 'UKOIL',
                          'wsj': '-',
                          'yahoo': '-',
                          },
               'US Oil': {'name': 'WTI (USD)',
                          'type': 'commodity',
                          'description': 'West Texas Intermetiate Oil option',
                          'axitrader': 'USOIL',
                          'wsj': '-',
                          'yahoo': '-',
                          },
               'Gold Option': {'name': 'Gold option (USD)',
                               'type': 'option',
                               'description': 'Gold option in USD',
                               'axitrader': 'GOLD.fs',
                               'wsj': '-',
                               'yahoo': '-',
                               },
               'Silver Option': {'name': 'Silver option (USD)',
                                 'type': 'option',
                                 'description': 'Silver option in USD',
                                 'axitrader': 'SILVER.fs',
                                 'wsj': '-',
                                 'yahoo': '-',
                                },
               'BRENT Option': {'name': 'Brent Oil option',
                                'type': 'option',
                                'description': 'Brent Oil Option',
                                'axitrader': 'BRENT.fs',
                                'wsj': '-',
                                'yahoo': '-',
                               },
               'WTI Option': {'name': 'WTI Oil option',
                              'type': 'option',
                              'description': 'West Texas Intermediate Oil option',
                              'axitrader': 'WTI.fs',
                              'wsj': '-',
                              'yahoo': '-',
                              },
               'Cocoa Option': {'name': 'Cocoa option (USD)',
                                'type': 'option',
                                'description': 'Cocoa option in USD',
                                'axitrader': 'COCOA.fs',
                                'wsj': '-',
                                'yahoo': '-',
                                },
               'Coffee Option': {'name': 'Coffee option (USD)',
                                 'type': 'option',
                                 'description': 'Coffee option in USD',
                                 'axitrader': 'COFFEE.fs',
                                 'wsj': '-',
                                 'yahoo': '-',
                                 },
               'Copper Option': {'name': 'COPPER option',
                                 'type': 'option',
                                 'description': 'Copper Option',
                                 'axitrader': 'COPPER.fs',
                                 'wsj': '-',
                                 'yahoo': '-',
                                 },
               'Natgas Option': {'name': 'Natural Gas option',
                                 'type': 'option',
                                 'description': 'Natural Gas option',
                                 'axitrader': 'NATGAS.fs',
                                 'wsj': '-',
                                 'yahoo': '-',
                                 },
               'Soybean Option': {'name': 'Soybean option',
                                 'type': 'option',
                                 'description': 'Soybean option',
                                 'axitrader': 'SOYBEAN.fs',
                                 'wsj': '-',
                                 'yahoo': '-',
                                 },
               }