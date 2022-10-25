Quickstart
==========

.. code:: ipython3

    import pandas as pd
    
    from parquetranger import TableRepo


.. code:: ipython3

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6],
            "B": ["x", "y", "z", "x1", "x2", "x3"],
            "C": [1, 2, 1, 1, 1, 2],
            "C2": ["a", "a", "b", "a", "c", "c"],
        },
        index=["a1", "a2", "a3", "a4", "a5", "a6"],
    )

.. code:: ipython3

    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>C2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a1</th>
          <td>1</td>
          <td>x</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a2</th>
          <td>2</td>
          <td>y</td>
          <td>2</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a3</th>
          <td>3</td>
          <td>z</td>
          <td>1</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>4</td>
          <td>x1</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a5</th>
          <td>5</td>
          <td>x2</td>
          <td>1</td>
          <td>c</td>
        </tr>
        <tr>
          <th>a6</th>
          <td>6</td>
          <td>x3</td>
          <td>2</td>
          <td>c</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    trepo = TableRepo("some_tmp_path", group_cols="C2")  # this creates the directory

.. code:: ipython3

    trepo.extend(df)

.. code:: ipython3

    trepo.get_full_df()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>C2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a1</th>
          <td>1</td>
          <td>x</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a2</th>
          <td>2</td>
          <td>y</td>
          <td>2</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>4</td>
          <td>x1</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a3</th>
          <td>3</td>
          <td>z</td>
          <td>1</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a5</th>
          <td>5</td>
          <td>x2</td>
          <td>1</td>
          <td>c</td>
        </tr>
        <tr>
          <th>a6</th>
          <td>6</td>
          <td>x3</td>
          <td>2</td>
          <td>c</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df2 = pd.DataFrame(
        {
            "A": [21, 22, 23],
            "B": ["X", "Y", "Z"],
            "C": [10,20,1],
            "C2": ["a", "b", "a"],
        },
        index=["a1", "a4", "a7"]
        )

.. code:: ipython3

    trepo.replace_records(df2)  # replaces based on index

.. code:: ipython3

    trepo.get_full_df()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>C2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a2</th>
          <td>2</td>
          <td>y</td>
          <td>2</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a1</th>
          <td>21</td>
          <td>X</td>
          <td>10</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a7</th>
          <td>23</td>
          <td>Z</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a3</th>
          <td>3</td>
          <td>z</td>
          <td>1</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>22</td>
          <td>Y</td>
          <td>20</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a5</th>
          <td>5</td>
          <td>x2</td>
          <td>1</td>
          <td>c</td>
        </tr>
        <tr>
          <th>a6</th>
          <td>6</td>
          <td>x3</td>
          <td>2</td>
          <td>c</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    trepo.replace_groups(df2)

.. code:: ipython3

    trepo.get_full_df()  # replaced the whole groups where C2==a and C2==b with the records that were present in df2




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>C2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a1</th>
          <td>21</td>
          <td>X</td>
          <td>10</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a7</th>
          <td>23</td>
          <td>Z</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>22</td>
          <td>Y</td>
          <td>20</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a5</th>
          <td>5</td>
          <td>x2</td>
          <td>1</td>
          <td>c</td>
        </tr>
        <tr>
          <th>a6</th>
          <td>6</td>
          <td>x3</td>
          <td>2</td>
          <td>c</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    trepo.replace_all(df2)  # erases everything and puts df2 in. all traces of df are lost

.. code:: ipython3

    trepo.get_full_df()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>C2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a1</th>
          <td>21</td>
          <td>X</td>
          <td>10</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a7</th>
          <td>23</td>
          <td>Z</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>22</td>
          <td>Y</td>
          <td>20</td>
          <td>b</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    trepo.replace_records(df, by_groups=True)  # replaces records based on index, but only looks for indices within groups, so this way duplicate a4 index is possible
    # as they are in different groups, with different values in C2

.. code:: ipython3

    trepo.get_full_df()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>C2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>a7</th>
          <td>23</td>
          <td>Z</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a1</th>
          <td>1</td>
          <td>x</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a2</th>
          <td>2</td>
          <td>y</td>
          <td>2</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>4</td>
          <td>x1</td>
          <td>1</td>
          <td>a</td>
        </tr>
        <tr>
          <th>a4</th>
          <td>22</td>
          <td>Y</td>
          <td>20</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a3</th>
          <td>3</td>
          <td>z</td>
          <td>1</td>
          <td>b</td>
        </tr>
        <tr>
          <th>a5</th>
          <td>5</td>
          <td>x2</td>
          <td>1</td>
          <td>c</td>
        </tr>
        <tr>
          <th>a6</th>
          <td>6</td>
          <td>x3</td>
          <td>2</td>
          <td>c</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    trepo.purge()  # deletes everything
