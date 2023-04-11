import pandas as pd
import pytest

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils

DATA = pd.DataFrame({'col': ['value', 'value', 'value2'],
                     'col2': ['value2', 'value', 'value3'],
                     'col3': ['value3', 'value4', 'value2']})


def test_select_columns_str_list():
    columns = ['col', 'col3']
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2'],
                                  'col3': ['value3', 'value4', 'value2']})
    output_data = pdutils.select_columns(DATA, columns)

    assert output_data.equals(expected_data)


def test_select_columns_str():
    column = 'col'
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2']})
    output_data = pdutils.select_columns(DATA, column)

    assert output_data.equals(expected_data)


def test_select_columns_int_list():
    columns = [0, 2]
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2'],
                                  'col3': ['value3', 'value4', 'value2']})
    output_data = pdutils.select_columns(DATA, columns)

    assert output_data.equals(expected_data)


def test_select_columns_int():
    column = 0
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2']})
    output_data = pdutils.select_columns(DATA, column)

    assert output_data.equals(expected_data)


def test_select_columns_int_str_list():
    columns = [0, 'col3']
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2'],
                                  'col3': ['value3', 'value4', 'value2']})
    output_data = pdutils.select_columns(DATA, columns)

    assert output_data.equals(expected_data)


def test_select_columns_reset_row_indexes():
    column = 'col'
    actual_data = pd.DataFrame({'col': ['value', 'value', 'value2'],
                                'col2': ['value2', 'value', 'value3'],
                                'col3': ['value3', 'value4', 'value2']},
                               index=['3', '5', '9'])
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2']})
    output_data = pdutils.select_columns(actual_data, column, reset_row_indexes=True)

    assert output_data.equals(expected_data)


def test_select_rows_list():
    rows = [0, 2]
    expected_data = pd.DataFrame(
        {'col': ['value', 'value2'], 'col2': ['value2', 'value3'],
         'col3': ['value3', 'value2']})
    output_data = pdutils.select_rows(DATA, rows)

    assert output_data.equals(expected_data)


def test_select_rows_int():
    row = 0
    expected_data = pd.DataFrame(
        {'col': ['value'], 'col2': ['value2'], 'col3': ['value3']})
    output_data = pdutils.select_rows(DATA, row)

    assert output_data.equals(expected_data)


def test_select_rows_by_value_column_name():
    column = 'col'
    value = 'value'
    expected_data = pd.DataFrame(
        {'col': ['value', 'value'], 'col2': ['value2', 'value'],
         'col3': ['value3', 'value4']})
    output_data = pdutils.select_rows_by_value(DATA, column, value)

    assert output_data.equals(expected_data)


def test_select_rows_by_value_column_index():
    column = 0
    value = 'value'
    expected_data = pd.DataFrame(
        {'col': ['value', 'value'], 'col2': ['value2', 'value'],
         'col3': ['value3', 'value4']})
    output_data = pdutils.select_rows_by_value(DATA, column, value)

    assert output_data.equals(expected_data)


def test_delete_columns_str_list():
    columns = ['col', 'col3']
    input_data = DATA.copy()
    expected_data = pd.DataFrame({'col2': ['value2', 'value', 'value3']})
    output_data = pdutils.delete_columns(input_data, columns)

    assert output_data.equals(expected_data)


def test_delete_columns_str():
    columns = 'col'
    input_data = DATA.copy()
    expected_data = pd.DataFrame({'col2': ['value2', 'value', 'value3'],
                                  'col3': ['value3', 'value4', 'value2']})
    output_data = pdutils.delete_columns(input_data, columns)

    assert output_data.equals(expected_data)


def test_delete_columns_int_list():
    columns = [0, 2]
    input_data = DATA.copy()
    expected_data = pd.DataFrame({'col2': ['value2', 'value', 'value3']})
    output_data = pdutils.delete_columns(input_data, columns)

    assert output_data.equals(expected_data)


def test_delete_columns_int():
    column = 0
    input_data = DATA.copy()
    expected_data = pd.DataFrame({'col2': ['value2', 'value', 'value3'],
                                  'col3': ['value3', 'value4', 'value2']})
    output_data = pdutils.delete_columns(input_data, column)

    assert output_data.equals(expected_data)


def test_delete_columns_str_int_list():
    columns = ['col', 2]
    input_data = DATA.copy()
    expected_data = pd.DataFrame({'col2': ['value2', 'value', 'value3']})
    output_data = pdutils.delete_columns(input_data, columns)

    assert output_data.equals(expected_data)


def test_select_value_row_column_str():
    row = 1
    column = 'col3'
    input_data = DATA.copy()
    expected_return = 'value4'
    actual_return = pdutils.select_value_row_column(input_data, row, column)
    assert expected_return == actual_return


def test_select_value_row_column_int():
    row = 1
    column = 2
    input_data = DATA.copy()
    expected_return = 'value4'
    actual_return = pdutils.select_value_row_column(input_data, row, column)
    assert expected_return == actual_return


def test_join_dataframes_x_wise():
    data_1 = DATA.copy()
    data_2 = pd.DataFrame({'col': ['a', 'd'], 'col2': ['b', 'e'],
                           'col3': ['c', 'f']})
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2', 'a',
                                          'd'],
                                  'col2': ['value2', 'value', 'value3', 'b',
                                           'e'],
                                  'col3': ['value3', 'value4', 'value2',
                                           'c', 'f']})
    output_data = pdutils.join_dataframes_x_wise([data_1, data_2])

    assert output_data.equals(expected_data)


def test_join_dataframes_x_wise_drop_column_names_true():
    data_1 = DATA.copy()
    data_2 = pd.DataFrame({'col': ['a', 'd'], 'col2': ['b', 'e'],
                           'col3': ['c', 'f']})
    expected_data = pd.DataFrame({'0': ['value', 'value', 'value2', 'a',
                                          'd'],
                                  '1': ['value2', 'value', 'value3', 'b',
                                           'e'],
                                  '2': ['value3', 'value4', 'value2',
                                           'c', 'f']})
    expected_data.columns = [''] * len(expected_data.columns)
    output_data = pdutils.join_dataframes_x_wise([data_1, data_2],
                                                 drop_column_names=True)

    assert output_data.equals(expected_data)


def test_join_dataframes_y_wise():
    data_1 = DATA.copy()
    data_2 = pd.DataFrame({'col3': ['value3', 'value4', 'value2'],
                           'col4': ['a', 'b', 'c'],
                           'col5': ['d', 'e', 'f']})
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2'],
                                  'col2': ['value2', 'value', 'value3'],
                                  'col3': ['value3', 'value4', 'value2'],
                                  'col4': ['a', 'b', 'c'],
                                  'col5': ['d', 'e', 'f']})
    output_data = pdutils.join_dataframes_y_wise([data_1, data_2])

    assert output_data.equals(expected_data)


def test_join_dataframes_y_wise_drop_duplicates_true():
    data_1 = DATA.copy()
    data_2 = pd.DataFrame({'col3': ['value3', 'value4', 'value2'],
                           'col4': ['a', 'b', 'c'],
                           'col5': ['d', 'e', 'f']})
    expected_data = pd.DataFrame({'col': ['value', 'value', 'value2'],
                                  'col2': ['value2', 'value', 'value3'],
                                  'col3': ['value3', 'value4', 'value2'],
                                  'col3_2': ['value3', 'value4', 'value2'],
                                  'col4': ['a', 'b', 'c'],
                                  'col5': ['d', 'e', 'f']})
    expected_data.columns = ['col', 'col2', 'col3', 'col3', 'col4', 'col5']
    output_data = pdutils.join_dataframes_y_wise([data_1, data_2], drop_duplicates=False)

    assert output_data.equals(expected_data)


def test_remove_duplicated_columns():
    dataset = pd.DataFrame({'col1': ['e', 'f', 'g'],
                            'col4': ['a', 'b', 'c'],
                            'col5': ['d', 'e', 'f'],
                            'col4': ['a', 'b', 'c']})
    expected_dataset = pd.DataFrame({'col1': ['e', 'f', 'g'],
                                     'col4': ['a', 'b', 'c'],
                                     'col5': ['d', 'e', 'f']})
    output_dataset = pdutils.remove_duplicated_columns(dataset)

    assert output_dataset.equals(expected_dataset)


def test_select_columns_with_substring():
    dataset = pd.DataFrame({'col1': ['e', 'f', 'g'],
                            'abc4': ['a', 'b', 'c'],
                            'col5': ['d', 'e', 'f'],
                            'abc2': ['i', 'j', 'k']})
    substring = "abc"
    expected_dataset = pd.DataFrame({'abc4': ['a', 'b', 'c'],
                                     'abc2': ['i', 'j', 'k']})
    output_dataset = pdutils.select_columns_with_substring(dataset, substring)

    assert output_dataset.equals(expected_dataset)


def test_select_columns_with_substring_more_than_one():
    dataset = pd.DataFrame({'col1': ['e', 'f', 'g'],
                            'abc4': ['a', 'b', 'c'],
                            'col5': ['d', 'e', 'f'],
                            'abc2': ['i', 'j', 'k']})
    substrings = ["col1", "abc"]
    expected_dataset = pd.DataFrame({'col1': ['e', 'f', 'g'],
                                     'abc4': ['a', 'b', 'c'],
                                     'abc2': ['i', 'j', 'k']})
    output_dataset = pdutils.select_columns_with_substring(dataset, substrings)

    assert output_dataset.equals(expected_dataset)


def test_add_row():
    expected_return = pd.DataFrame({'col': ['value', 'value', 'value2', 'a'],
                                    'col2': ['value2', 'value', 'value3', 'b'],
                                    'col3': ['value3', 'value4', 'value2', 'c']})
    values = ['a', 'b', 'c']
    actual_return = pdutils.add_row(DATA, values)

    assert expected_return.equals(actual_return)


def test_dataframe_to_list_multicolumn_data():
    dataset = pd.DataFrame({'col1': ['e', 'f', 'g'],
                            'abc4': ['a', 'b', 'c'],
                            'col5': ['d', 'e', 'f'],
                            'abc2': ['i', 'j', 'k']})

    with pytest.raises(ValueError) as e_info:
        pdutils.dataframe_to_list(dataset)
    assert str(e_info.value) == "Parameter dataframe must have one single column"


def test_dataframe_to_list():
    dataset = pd.DataFrame({'col1': ['e', 'f', 'g']})
    expected_return = ['e', 'f', 'g']
    actual_return = pdutils.dataframe_to_list(dataset)

    assert expected_return == actual_return


def test_add_prefix_to_column_names():
    expected_return = DATA.copy()
    expected_return.columns = ["a_col", "a_col2", "a_col3"]
    actual_return = pdutils.add_prefix_to_column_names(DATA, "a_")

    assert expected_return.equals(actual_return)


def test_join_inner_by_date_no_date_dataframe_0():
    data_2 = pd.DataFrame({dscfg.DATE_COLUMN_NAME: ['1', '2', '3'],
                           'col': ['value', 'value', 'value2'],
                           'col2': ['value2', 'value', 'value3'],
                           'col3': ['value3', 'value4', 'value2']})
    with pytest.raises(ValueError) as e_info:
        pdutils.join_inner_by_date([DATA, data_2])
    assert str(e_info.value) == "Parameter dataframe 0 does not contain date column"


def test_join_inner_by_date_no_date_dataframe_2():
    data_1 = pd.DataFrame({dscfg.DATE_COLUMN_NAME: ['1', '2', '3'],
                           'col': ['value', 'value', 'value2'],
                           'col2': ['value2', 'value', 'value3'],
                           'col3': ['value3', 'value4', 'value2']})
    with pytest.raises(ValueError) as e_info:
        pdutils.join_inner_by_date([data_1, DATA])
    assert str(e_info.value) == "Parameter dataframe 1 does not contain date column"


def test_join_inner_by_date():
    data_1 = pd.DataFrame({dscfg.DATE_COLUMN_NAME: ['1', '2', '3'],
                           'col': ['value', 'value', 'value2'],
                           'col2': ['value2', 'value', 'value3']})
    data_2 = pd.DataFrame({dscfg.DATE_COLUMN_NAME: ['1', '3', '4'],
                           'col3': ['a', 'b', 'c']})
    expected_output = pd.DataFrame({dscfg.DATE_COLUMN_NAME: ['1', '3'],
                                    'col': ['value', 'value2'],
                                    'col2': ['value2', 'value3'],
                                    'col3': ['a', 'b']})
    actual_output = pdutils.join_inner_by_date([data_1, data_2])

    assert expected_output.equals(actual_output)
