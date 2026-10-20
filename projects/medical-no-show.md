---
layout: single
permalink: /projects/medical-no-show/
title: "Medical Appointment No-Show Prediction"
classes: wide
author_profile: false
share: false
related: false
---
<div class="notebook-intro">
<p class="notebook-kicker">PROJECT NOTEBOOK · ORIGINAL CONTENT</p>
<p>Below is the original notebook content, displayed as a continuous reading page. The Markdown, code, saved outputs, and cell order have not been rewritten.</p>
<div class="notebook-actions">
<a href="{{ '/assets/notebooks/Medical_Appointment_No_Show_Prediction.ipynb' | relative_url }}" download>Download original notebook (.ipynb) ↧</a>
<a href="{{ '/assets/notebooks/Medical_Appointment_No_Show_Prediction.html' | relative_url }}" target="_blank" rel="noopener">Open original HTML export ↗</a>
<a href="{{ '/projects/' | relative_url }}">All projects ←</a>
</div></div>

<div class="notebook-reading">
<div class="container" id="notebook-container">
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Predicting-Medical-Appointment-No-Shows-Using-SQL-and-Machine-Learning"><strong>Predicting Medical Appointment No-Shows Using SQL and Machine Learning</strong><a class="anchor-link" href="#Predicting-Medical-Appointment-No-Shows-Using-SQL-and-Machine-Learning">¶</a></h1><p>Tools: SQL, Neon PostgreSQL, Python, pandas, NumPy, seaborn, matplotlib, scikit-learn</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Table-of-Contents">Table of Contents<a class="anchor-link" href="#Table-of-Contents">¶</a></h2><ol>
<li>Project Overview</li>
<li>Dataset &amp; Database Setup</li>
<li>SQL Data Exploration</li>
<li>Exploratory Data Analysis (EDA) in Python</li>
<li>Feature Engineering</li>
<li>Feature Analysis</li>
<li>Machine Learning Model Development</li>
<li>Final Model Evaluation</li>
<li>Model Interpretation</li>
<li>Conclusion</li>
</ol>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="1.-Project-Overview">1. Project Overview<a class="anchor-link" href="#1.-Project-Overview">¶</a></h2><p>Medical appointment no-shows can result in unused appointment slots and inefficient use of healthcare resources. In this project, I analyzed patient and appointment data to identify patterns associated with missed appointments and developed a machine learning model to predict whether an appointment would result in a show or no-show.</p>
<h3 id="Project-Goal">Project Goal<a class="anchor-link" href="#Project-Goal">¶</a></h3><p>The goal of this project was to go beyond exploratory analysis and build a binary classification model that predicts whether a medical appointment will result in a no-show using patient and appointment information available before the appointment outcome.</p>
<p>I defined the prediction target as:</p>
<ul>
<li>Show = 0</li>
<li>No-show = 1</li>
</ul>
<p>I used SQL for initial data exploration and Python for exploratory analysis, feature engineering, and machine learning. Because no-show appointments represent the minority class, I evaluated the models using accuracy, precision, recall, F1-score, and ROC-AUC, with particular attention to the model's ability to identify actual no-show appointments.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="2.-Dataset-&amp;-Database-Setup">2. Dataset &amp; Database Setup<a class="anchor-link" href="#2.-Dataset-&amp;-Database-Setup">¶</a></h2><p>I used the Medical Appointment No Shows dataset from Kaggle, which contains patient characteristics and appointment information such as age, gender, scheduling dates, SMS receipt, health-related indicators, neighbourhood, and appointment outcome.</p>
<p>The original dataset contained 110,527 appointment records and 14 variables.</p>
<p>I downloaded the original CSV file from Kaggle and loaded its records into Neon PostgreSQL as the <code>medical_appointments</code> table. After the initial database setup, I retrieved the data directly from PostgreSQL for SQL exploration and subsequent Python analysis rather than repeatedly reading the local CSV file.</p>
<p>A SQL validation query confirmed that all 110,527 appointment records were successfully stored in the database.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Variables-Used-in-the-Analysis">Variables Used in the Analysis<a class="anchor-link" href="#Variables-Used-in-the-Analysis">¶</a></h3><p>The dataset contains patient characteristics, appointment information, and the final appointment outcome. I also created several additional variables from the original data for analysis and machine learning.</p>
<table>
<thead>
<tr>
<th>Variable</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>age</code></td>
<td>Patient age</td>
</tr>
<tr>
<td><code>gender</code></td>
<td>Patient gender</td>
</tr>
<tr>
<td><code>neighbourhood</code></td>
<td>Neighbourhood associated with the appointment</td>
</tr>
<tr>
<td><code>scholarship</code></td>
<td>Whether the patient was enrolled in the Brazilian Bolsa Família welfare program</td>
</tr>
<tr>
<td><code>hypertension</code></td>
<td>Whether the patient had hypertension</td>
</tr>
<tr>
<td><code>diabetes</code></td>
<td>Whether the patient had diabetes</td>
</tr>
<tr>
<td><code>alcoholism</code></td>
<td>Whether alcoholism was recorded</td>
</tr>
<tr>
<td><code>handicap</code></td>
<td>Recorded disability-related indicator</td>
</tr>
<tr>
<td><code>sms_received</code></td>
<td>Whether the patient received an SMS reminder</td>
</tr>
<tr>
<td><code>waiting_days</code></td>
<td>Number of days between scheduling and the appointment</td>
</tr>
<tr>
<td><code>scheduled_weekday</code></td>
<td>Day of the week when the appointment was scheduled</td>
</tr>
<tr>
<td><code>appointment_weekday</code></td>
<td>Day of the week of the appointment</td>
</tr>
<tr>
<td><code>no_show_binary</code></td>
<td>Binary prediction target: <strong>0 = Show, 1 = No-show</strong></td>
</tr>
</tbody>
</table>
<p><code>waiting_days</code>, <code>scheduled_weekday</code>, <code>appointment_weekday</code>, and <code>no_show_binary</code> were created from the original appointment information during feature engineering.</p>
<p>I excluded <code>patient_id</code> and <code>appointment_id</code> from the machine learning models because they function primarily as identifiers rather than meaningful predictive features.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="err">!</span><span class="n">pip</span> <span class="n">install</span> <span class="o">-</span><span class="n">q</span> <span class="n">sqlalchemy</span> <span class="s2">"psycopg[binary]"</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="kn">import</span><span class="w"> </span><span class="nn">pandas</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">pd</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">seaborn</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">sns</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">IPython.display</span><span class="w"> </span><span class="kn">import</span> <span class="n">display</span><span class="p">,</span> <span class="n">Markdown</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">getpass</span><span class="w"> </span><span class="kn">import</span> <span class="n">getpass</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sqlalchemy</span><span class="w"> </span><span class="kn">import</span> <span class="n">create_engine</span><span class="p">,</span> <span class="n">text</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.model_selection</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">train_test_split</span><span class="p">,</span>
    <span class="n">StratifiedKFold</span><span class="p">,</span>
    <span class="n">cross_val_predict</span>
<span class="p">)</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.pipeline</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pipeline</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.preprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">StandardScaler</span><span class="p">,</span> <span class="n">OneHotEncoder</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.compose</span><span class="w"> </span><span class="kn">import</span> <span class="n">ColumnTransformer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.inspection</span><span class="w"> </span><span class="kn">import</span> <span class="n">permutation_importance</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.linear_model</span><span class="w"> </span><span class="kn">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.ensemble</span><span class="w"> </span><span class="kn">import</span> <span class="n">RandomForestClassifier</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.metrics</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">accuracy_score</span><span class="p">,</span>
    <span class="n">precision_score</span><span class="p">,</span>
    <span class="n">recall_score</span><span class="p">,</span>
    <span class="n">f1_score</span><span class="p">,</span>
    <span class="n">roc_auc_score</span><span class="p">,</span>
    <span class="n">confusion_matrix</span><span class="p">,</span>
    <span class="n">ConfusionMatrixDisplay</span><span class="p">,</span>
    <span class="n">RocCurveDisplay</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="k">def</span><span class="w"> </span><span class="nf">display_table</span><span class="p">(</span><span class="n">df</span><span class="p">):</span>
    <span class="n">display_df</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

    <span class="c1"># Make column names easier to read</span>
    <span class="n">column_names</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"patient_id"</span><span class="p">:</span> <span class="s2">"Patient ID"</span><span class="p">,</span>
        <span class="s2">"appointment_id"</span><span class="p">:</span> <span class="s2">"Appointment ID"</span><span class="p">,</span>
        <span class="s2">"gender"</span><span class="p">:</span> <span class="s2">"Gender"</span><span class="p">,</span>
        <span class="s2">"scheduled_day"</span><span class="p">:</span> <span class="s2">"Scheduled Day"</span><span class="p">,</span>
        <span class="s2">"appointment_day"</span><span class="p">:</span> <span class="s2">"Appointment Day"</span><span class="p">,</span>
        <span class="s2">"age"</span><span class="p">:</span> <span class="s2">"Age"</span><span class="p">,</span>
        <span class="s2">"neighbourhood"</span><span class="p">:</span> <span class="s2">"Neighbourhood"</span><span class="p">,</span>
        <span class="s2">"scholarship"</span><span class="p">:</span> <span class="s2">"Scholarship"</span><span class="p">,</span>
        <span class="s2">"hypertension"</span><span class="p">:</span> <span class="s2">"Hypertension"</span><span class="p">,</span>
        <span class="s2">"diabetes"</span><span class="p">:</span> <span class="s2">"Diabetes"</span><span class="p">,</span>
        <span class="s2">"alcoholism"</span><span class="p">:</span> <span class="s2">"Alcoholism"</span><span class="p">,</span>
        <span class="s2">"handicap"</span><span class="p">:</span> <span class="s2">"Handicap"</span><span class="p">,</span>
        <span class="s2">"sms_received"</span><span class="p">:</span> <span class="s2">"SMS Received"</span><span class="p">,</span>
        <span class="s2">"no_show"</span><span class="p">:</span> <span class="s2">"Appointment Outcome"</span><span class="p">,</span>
        <span class="s2">"appointment_count"</span><span class="p">:</span> <span class="s2">"Appointment Count"</span><span class="p">,</span>
        <span class="s2">"no_show_count"</span><span class="p">:</span> <span class="s2">"No-show Count"</span><span class="p">,</span>
        <span class="s2">"no_show_rate"</span><span class="p">:</span> <span class="s2">"No-show Rate (%)"</span><span class="p">,</span>
        <span class="s2">"percentage"</span><span class="p">:</span> <span class="s2">"Percentage (%)"</span><span class="p">,</span>
        <span class="s2">"waiting_days"</span><span class="p">:</span> <span class="s2">"Waiting Days"</span><span class="p">,</span>
        <span class="s2">"waiting_group"</span><span class="p">:</span> <span class="s2">"Waiting Group"</span><span class="p">,</span>
        <span class="s2">"min_age"</span><span class="p">:</span> <span class="s2">"Minimum Age"</span><span class="p">,</span>
        <span class="s2">"max_age"</span><span class="p">:</span> <span class="s2">"Maximum Age"</span><span class="p">,</span>
        <span class="s2">"negative_age_count"</span><span class="p">:</span> <span class="s2">"Negative Age Count"</span><span class="p">,</span>
        <span class="s2">"age_group"</span><span class="p">:</span> <span class="s2">"Age Group"</span><span class="p">,</span>
        <span class="s2">"no_show_binary"</span><span class="p">:</span> <span class="s2">"No-show Binary"</span><span class="p">,</span>
        <span class="s2">"scheduled_weekday"</span><span class="p">:</span> <span class="s2">"Scheduled Weekday"</span><span class="p">,</span>
        <span class="s2">"appointment_weekday"</span><span class="p">:</span> <span class="s2">"Appointment Weekday"</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">display_df</span> <span class="o">=</span> <span class="n">display_df</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">column_names</span><span class="p">)</span>

    <span class="k">if</span> <span class="s2">"Patient ID"</span> <span class="ow">in</span> <span class="n">display_df</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
        <span class="n">display_df</span><span class="p">[</span><span class="s2">"Patient ID"</span><span class="p">]</span> <span class="o">=</span> <span class="n">display_df</span><span class="p">[</span><span class="s2">"Patient ID"</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">"int64"</span><span class="p">)</span>

    <span class="k">if</span> <span class="s2">"Appointment Outcome"</span> <span class="ow">in</span> <span class="n">display_df</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
        <span class="n">display_df</span><span class="p">[</span><span class="s2">"Appointment Outcome"</span><span class="p">]</span> <span class="o">=</span> <span class="n">display_df</span><span class="p">[</span>
            <span class="s2">"Appointment Outcome"</span>
        <span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">({</span>
            <span class="s2">"No"</span><span class="p">:</span> <span class="s2">"Show"</span><span class="p">,</span>
            <span class="s2">"Yes"</span><span class="p">:</span> <span class="s2">"No-show"</span>
        <span class="p">})</span>
    <span class="n">binary_columns</span> <span class="o">=</span> <span class="p">[</span>
        <span class="s2">"SMS Received"</span><span class="p">,</span>
        <span class="s2">"Scholarship"</span><span class="p">,</span>
        <span class="s2">"Hypertension"</span><span class="p">,</span>
        <span class="s2">"Diabetes"</span><span class="p">,</span>
        <span class="s2">"Alcoholism"</span>
    <span class="p">]</span>

    <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="n">binary_columns</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">column</span> <span class="ow">in</span> <span class="n">display_df</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
            <span class="n">display_df</span><span class="p">[</span><span class="n">column</span><span class="p">]</span> <span class="o">=</span> <span class="n">display_df</span><span class="p">[</span><span class="n">column</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">({</span>
                <span class="mi">0</span><span class="p">:</span> <span class="s2">"No"</span><span class="p">,</span>
                <span class="mi">1</span><span class="p">:</span> <span class="s2">"Yes"</span><span class="p">,</span>
                <span class="s2">"0"</span><span class="p">:</span> <span class="s2">"No"</span><span class="p">,</span>
                <span class="s2">"1"</span><span class="p">:</span> <span class="s2">"Yes"</span>
            <span class="p">})</span>

    <span class="k">return</span> <span class="n">display_df</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">DATABASE_URL</span> <span class="o">=</span> <span class="n">getpass</span><span class="p">(</span><span class="s2">"Paste your Neon connection string: "</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Paste your Neon connection string: ··········
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="k">if</span> <span class="n">DATABASE_URL</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"postgresql://"</span><span class="p">):</span>
    <span class="n">DATABASE_URL</span> <span class="o">=</span> <span class="n">DATABASE_URL</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span>
        <span class="s2">"postgresql://"</span><span class="p">,</span>
        <span class="s2">"postgresql+psycopg://"</span><span class="p">,</span>
        <span class="mi">1</span>
    <span class="p">)</span>

<span class="n">engine</span> <span class="o">=</span> <span class="n">create_engine</span><span class="p">(</span><span class="n">DATABASE_URL</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="k">with</span> <span class="n">engine</span><span class="o">.</span><span class="n">connect</span><span class="p">()</span> <span class="k">as</span> <span class="n">conn</span><span class="p">:</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">conn</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">text</span><span class="p">(</span><span class="s2">"SELECT 1"</span><span class="p">))</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">scalar</span><span class="p">())</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>1
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT COUNT(*) AS total_appointments</span>
<span class="s2">FROM medical_appointments;</span>
<span class="s2">"""</span>

<span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-b81c65ab-2113-4510-939c-22b4345dbfee">
<div>
<style scoped="">
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
<th>total_appointments</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>110527</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-b81c65ab-2113-4510-939c-22b4345dbfee')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-b81c65ab-2113-4510-939c-22b4345dbfee button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b81c65ab-2113-4510-939c-22b4345dbfee');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT *</span>
<span class="s2">FROM medical_appointments;</span>
<span class="s2">"""</span>

<span class="n">df_sql</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">df_sql</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-0123cae1-40b9-4e94-971b-110b1cd644d7">
<div>
<style scoped="">
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
<th>Patient ID</th>
<th>Appointment ID</th>
<th>Gender</th>
<th>Scheduled Day</th>
<th>Appointment Day</th>
<th>Age</th>
<th>Neighbourhood</th>
<th>Scholarship</th>
<th>Hypertension</th>
<th>Diabetes</th>
<th>Alcoholism</th>
<th>Handicap</th>
<th>SMS Received</th>
<th>Appointment Outcome</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>29872499824296</td>
<td>5642903</td>
<td>F</td>
<td>2016-04-29T18:38:08Z</td>
<td>2016-04-29T00:00:00Z</td>
<td>62</td>
<td>JARDIM DA PENHA</td>
<td>No</td>
<td>Yes</td>
<td>No</td>
<td>No</td>
<td>0</td>
<td>No</td>
<td>Show</td>
</tr>
<tr>
<th>1</th>
<td>558997776694438</td>
<td>5642503</td>
<td>M</td>
<td>2016-04-29T16:08:27Z</td>
<td>2016-04-29T00:00:00Z</td>
<td>56</td>
<td>JARDIM DA PENHA</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>0</td>
<td>No</td>
<td>Show</td>
</tr>
<tr>
<th>2</th>
<td>4262962299951</td>
<td>5642549</td>
<td>F</td>
<td>2016-04-29T16:19:04Z</td>
<td>2016-04-29T00:00:00Z</td>
<td>62</td>
<td>MATA DA PRAIA</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>0</td>
<td>No</td>
<td>Show</td>
</tr>
<tr>
<th>3</th>
<td>867951213174</td>
<td>5642828</td>
<td>F</td>
<td>2016-04-29T17:29:31Z</td>
<td>2016-04-29T00:00:00Z</td>
<td>8</td>
<td>PONTAL DE CAMBURI</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>0</td>
<td>No</td>
<td>Show</td>
</tr>
<tr>
<th>4</th>
<td>8841186448183</td>
<td>5642494</td>
<td>F</td>
<td>2016-04-29T16:07:23Z</td>
<td>2016-04-29T00:00:00Z</td>
<td>56</td>
<td>JARDIM DA PENHA</td>
<td>No</td>
<td>Yes</td>
<td>Yes</td>
<td>No</td>
<td>0</td>
<td>No</td>
<td>Show</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-0123cae1-40b9-4e94-971b-110b1cd644d7')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-0123cae1-40b9-4e94-971b-110b1cd644d7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0123cae1-40b9-4e94-971b-110b1cd644d7');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="3.-SQL-Data-Exploration">3. SQL Data Exploration<a class="anchor-link" href="#3.-SQL-Data-Exploration">¶</a></h2><p>Before moving into Python analysis, I used SQL to examine the appointment outcome distribution and explore how selected patient and appointment characteristics were associated with no-show behavior.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT</span>
<span class="s2">    no_show,</span>
<span class="s2">    COUNT(*) AS appointment_count,</span>
<span class="s2">    ROUND(</span>
<span class="s2">        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),</span>
<span class="s2">        2</span>
<span class="s2">    ) AS percentage</span>
<span class="s2">FROM medical_appointments</span>
<span class="s2">GROUP BY no_show</span>
<span class="s2">ORDER BY no_show;</span>
<span class="s2">"""</span>

<span class="n">no_show_distribution</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">no_show_distribution</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-1b3bcaac-4971-4df6-8ce3-d29311355048">
<div>
<style scoped="">
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
<th>Appointment Outcome</th>
<th>Appointment Count</th>
<th>Percentage (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Show</td>
<td>88208</td>
<td>79.81</td>
</tr>
<tr>
<th>1</th>
<td>No-show</td>
<td>22319</td>
<td>20.19</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-1b3bcaac-4971-4df6-8ce3-d29311355048')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-1b3bcaac-4971-4df6-8ce3-d29311355048 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1b3bcaac-4971-4df6-8ce3-d29311355048');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Target-Definition">Target Definition<a class="anchor-link" href="#Target-Definition">¶</a></h3><p>In the original dataset, the <code>no_show</code> variable uses "Yes" to indicate a no-show and "No" to indicate attendance. For readability, I display these outcomes as No-show and Show throughout the analysis.</p>
<p>For machine learning, I converted the appointment outcome into a binary target:</p>
<ul>
<li>No-show = 1</li>
<li>Show = 0</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Appointment-Outcome-Distribution">Key Finding: Appointment Outcome Distribution<a class="anchor-link" href="#Key-Finding:-Appointment-Outcome-Distribution">¶</a></h3><p>I found that <strong>20.19%</strong> of the appointments resulted in a no-show, while <strong>79.81%</strong> resulted in attendance.</p>
<p>Because no-show cases represent the smaller class, I will later use stratified sampling and evaluate the classification models with particular attention to <strong>no-show recall</strong>, rather than relying on accuracy alone.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT</span>
<span class="s2">    sms_received,</span>
<span class="s2">    COUNT(*) AS appointment_count,</span>
<span class="s2">    SUM(CASE WHEN no_show = 'Yes' THEN 1 ELSE 0 END) AS no_show_count,</span>
<span class="s2">    ROUND(</span>
<span class="s2">        AVG(CASE WHEN no_show = 'Yes' THEN 1.0 ELSE 0 END) * 100,</span>
<span class="s2">        2</span>
<span class="s2">    ) AS no_show_rate</span>
<span class="s2">FROM medical_appointments</span>
<span class="s2">GROUP BY sms_received</span>
<span class="s2">ORDER BY sms_received;</span>
<span class="s2">"""</span>

<span class="n">sms_result</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">sms_result</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-061cde05-23fb-4de1-abd9-ffe68e67eba3">
<div>
<style scoped="">
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
<th>SMS Received</th>
<th>Appointment Count</th>
<th>No-show Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>No</td>
<td>75045</td>
<td>12535</td>
<td>16.70</td>
</tr>
<tr>
<th>1</th>
<td>Yes</td>
<td>35482</td>
<td>9784</td>
<td>27.57</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-061cde05-23fb-4de1-abd9-ffe68e67eba3')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-061cde05-23fb-4de1-abd9-ffe68e67eba3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-061cde05-23fb-4de1-abd9-ffe68e67eba3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-SMS-Receipt-and-No-Show-Rate">Key Finding: SMS Receipt and No-Show Rate<a class="anchor-link" href="#Key-Finding:-SMS-Receipt-and-No-Show-Rate">¶</a></h3><p>I found that appointments with an SMS reminder had a higher no-show rate (27.57%) than appointments without an SMS reminder (16.70%).
This result shows an association between SMS receipt and no-show behavior in this dataset, but it does not mean that receiving an SMS caused patients to miss their appointments.</p>
<p>Moreover, SMS reminders were not distributed evenly across waiting periods, and longer waiting periods were also associated with higher no-show rates. Therefore, the overall difference should not be interpreted as an effect of SMS receipt alone.</p>
<p>To examine this relationship more carefully, I further compared no-show rates between appointments with and without SMS reminders within similar waiting-day groups.This additional analysis helps separate the relationship between SMS receipt and no-show behavior from the effect of longer waiting periods.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">WITH appointment_waiting AS (</span>
<span class="s2">    SELECT</span>
<span class="s2">        sms_received,</span>
<span class="s2">        no_show,</span>
<span class="s2">        appointment_day::timestamptz::date</span>
<span class="s2">            - scheduled_day::timestamptz::date AS waiting_days</span>
<span class="s2">    FROM medical_appointments</span>
<span class="s2">),</span>

<span class="s2">waiting_groups AS (</span>
<span class="s2">    SELECT</span>
<span class="s2">        sms_received,</span>
<span class="s2">        no_show,</span>
<span class="s2">        CASE</span>
<span class="s2">            WHEN waiting_days BETWEEN 1 AND 7 THEN '1-7 Days'</span>
<span class="s2">            WHEN waiting_days BETWEEN 8 AND 30 THEN '8-30 Days'</span>
<span class="s2">            WHEN waiting_days &gt;= 31 THEN '31+ Days'</span>
<span class="s2">        END AS waiting_group</span>
<span class="s2">    FROM appointment_waiting</span>
<span class="s2">    WHERE waiting_days &gt;= 1</span>
<span class="s2">)</span>

<span class="s2">SELECT</span>
<span class="s2">    waiting_group,</span>
<span class="s2">    sms_received,</span>
<span class="s2">    COUNT(*) AS appointment_count,</span>
<span class="s2">    SUM(</span>
<span class="s2">        CASE WHEN no_show = 'Yes' THEN 1 ELSE 0 END</span>
<span class="s2">    ) AS no_show_count,</span>
<span class="s2">    ROUND(</span>
<span class="s2">        AVG(</span>
<span class="s2">            CASE WHEN no_show = 'Yes' THEN 1.0 ELSE 0 END</span>
<span class="s2">        ) * 100,</span>
<span class="s2">        2</span>
<span class="s2">    ) AS no_show_rate</span>

<span class="s2">FROM waiting_groups</span>

<span class="s2">WHERE waiting_group IS NOT NULL</span>

<span class="s2">GROUP BY</span>
<span class="s2">    waiting_group,</span>
<span class="s2">    sms_received</span>

<span class="s2">ORDER BY</span>
<span class="s2">    CASE</span>
<span class="s2">        WHEN waiting_group = '1-7 Days' THEN 1</span>
<span class="s2">        WHEN waiting_group = '8-30 Days' THEN 2</span>
<span class="s2">        WHEN waiting_group = '31+ Days' THEN 3</span>
<span class="s2">    END,</span>
<span class="s2">    sms_received;</span>
<span class="s2">"""</span>

<span class="n">sms_waiting_result</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">sms_waiting_result</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-1450bd11-ea51-4b08-952c-935ee96b364b">
<div>
<style scoped="">
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
<th>Waiting Group</th>
<th>SMS Received</th>
<th>Appointment Count</th>
<th>No-show Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>1-7 Days</td>
<td>No</td>
<td>20637</td>
<td>5028</td>
<td>24.36</td>
</tr>
<tr>
<th>1</th>
<td>1-7 Days</td>
<td>Yes</td>
<td>11548</td>
<td>2744</td>
<td>23.76</td>
</tr>
<tr>
<th>2</th>
<td>8-30 Days</td>
<td>No</td>
<td>11872</td>
<td>4221</td>
<td>35.55</td>
</tr>
<tr>
<th>3</th>
<td>8-30 Days</td>
<td>Yes</td>
<td>17524</td>
<td>5104</td>
<td>29.13</td>
</tr>
<tr>
<th>4</th>
<td>31+ Days</td>
<td>No</td>
<td>3968</td>
<td>1489</td>
<td>37.53</td>
</tr>
<tr>
<th>5</th>
<td>31+ Days</td>
<td>Yes</td>
<td>6410</td>
<td>1936</td>
<td>30.20</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-1450bd11-ea51-4b08-952c-935ee96b364b')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-1450bd11-ea51-4b08-952c-935ee96b364b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1450bd11-ea51-4b08-952c-935ee96b364b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-SMS-and-No-Show-Rates-Within-Waiting-Periods">Key Finding: SMS and No-Show Rates Within Waiting Periods<a class="anchor-link" href="#Key-Finding:-SMS-and-No-Show-Rates-Within-Waiting-Periods">¶</a></h3><p>After comparing appointments within the same waiting-period groups, I found that appointments receiving an SMS reminder consistently had lower no-show rates than appointments without an SMS reminder.</p>
<p>For appointments scheduled 1–7 days in advance, the no-show rate was 23.76% with SMS compared with 24.36% without SMS. The difference became larger for longer waiting periods: 29.13% vs. 35.55% for 8–30 days and 30.20% vs. 37.53% for 31+ days.</p>
<p>This result suggests that the higher overall no-show rate previously observed among SMS recipients was partly related to differences in waiting periods. Within comparable waiting periods, SMS receipt was associated with lower no-show rates. However, because this is observational data, I do not interpret this relationship as evidence that SMS reminders directly caused the reduction in no-shows.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">WITH appointment_waiting AS (</span>
<span class="s2">    SELECT</span>
<span class="s2">        no_show,</span>
<span class="s2">        appointment_day::timestamptz::date</span>
<span class="s2">            - scheduled_day::timestamptz::date AS waiting_days</span>
<span class="s2">    FROM medical_appointments</span>
<span class="s2">),</span>

<span class="s2">waiting_groups AS (</span>
<span class="s2">    SELECT</span>
<span class="s2">        no_show,</span>
<span class="s2">        waiting_days,</span>
<span class="s2">        CASE</span>
<span class="s2">            WHEN waiting_days &lt; 0 THEN 'Invalid (&lt;0 Days)'</span>
<span class="s2">            WHEN waiting_days = 0 THEN 'Same Day'</span>
<span class="s2">            WHEN waiting_days BETWEEN 1 AND 7 THEN '1-7 Days'</span>
<span class="s2">            WHEN waiting_days BETWEEN 8 AND 30 THEN '8-30 Days'</span>
<span class="s2">            ELSE '31+ Days'</span>
<span class="s2">        END AS waiting_group</span>
<span class="s2">    FROM appointment_waiting</span>
<span class="s2">)</span>

<span class="s2">SELECT</span>
<span class="s2">    waiting_group,</span>
<span class="s2">    COUNT(*) AS appointment_count,</span>

<span class="s2">    SUM(</span>
<span class="s2">        CASE WHEN no_show = 'Yes' THEN 1 ELSE 0 END</span>
<span class="s2">    ) AS no_show_count,</span>

<span class="s2">    ROUND(</span>
<span class="s2">        AVG(</span>
<span class="s2">            CASE WHEN no_show = 'Yes' THEN 1.0 ELSE 0 END</span>
<span class="s2">        ) * 100,</span>
<span class="s2">        2</span>
<span class="s2">    ) AS no_show_rate</span>

<span class="s2">FROM waiting_groups</span>

<span class="s2">GROUP BY waiting_group</span>

<span class="s2">ORDER BY MIN(waiting_days);</span>
<span class="s2">"""</span>

<span class="n">waiting_result</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">waiting_result</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-5fe51e48-01e4-489e-8096-f1ce54436431">
<div>
<style scoped="">
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
<th>Waiting Group</th>
<th>Appointment Count</th>
<th>No-show Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Invalid (&lt;0 Days)</td>
<td>5</td>
<td>5</td>
<td>100.00</td>
</tr>
<tr>
<th>1</th>
<td>Same Day</td>
<td>38563</td>
<td>1792</td>
<td>4.65</td>
</tr>
<tr>
<th>2</th>
<td>1-7 Days</td>
<td>32185</td>
<td>7772</td>
<td>24.15</td>
</tr>
<tr>
<th>3</th>
<td>8-30 Days</td>
<td>29396</td>
<td>9325</td>
<td>31.72</td>
</tr>
<tr>
<th>4</th>
<td>31+ Days</td>
<td>10378</td>
<td>3425</td>
<td>33.00</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-5fe51e48-01e4-489e-8096-f1ce54436431')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-5fe51e48-01e4-489e-8096-f1ce54436431 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5fe51e48-01e4-489e-8096-f1ce54436431');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT</span>
<span class="s2">    patient_id,</span>
<span class="s2">    appointment_id,</span>
<span class="s2">    scheduled_day,</span>
<span class="s2">    appointment_day,</span>
<span class="s2">    appointment_day::timestamptz::date</span>
<span class="s2">        - scheduled_day::timestamptz::date AS waiting_days,</span>
<span class="s2">    no_show</span>
<span class="s2">FROM medical_appointments</span>
<span class="s2">WHERE</span>
<span class="s2">    appointment_day::timestamptz::date</span>
<span class="s2">        - scheduled_day::timestamptz::date &lt; 0</span>
<span class="s2">ORDER BY waiting_days;</span>
<span class="s2">"""</span>

<span class="n">invalid_waiting</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">invalid_waiting</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-9d7041a5-a9c0-4301-94ad-715f7d9edaee">
<div>
<style scoped="">
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
<th>Patient ID</th>
<th>Appointment ID</th>
<th>Scheduled Day</th>
<th>Appointment Day</th>
<th>Waiting Days</th>
<th>Appointment Outcome</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>998231581612122</td>
<td>5686628</td>
<td>2016-05-11T13:49:20Z</td>
<td>2016-05-05T00:00:00Z</td>
<td>-6</td>
<td>No-show</td>
</tr>
<tr>
<th>1</th>
<td>7839272661752</td>
<td>5679978</td>
<td>2016-05-10T10:51:53Z</td>
<td>2016-05-09T00:00:00Z</td>
<td>-1</td>
<td>No-show</td>
</tr>
<tr>
<th>2</th>
<td>7896293967868</td>
<td>5715660</td>
<td>2016-05-18T14:50:41Z</td>
<td>2016-05-17T00:00:00Z</td>
<td>-1</td>
<td>No-show</td>
</tr>
<tr>
<th>3</th>
<td>24252258389979</td>
<td>5664962</td>
<td>2016-05-05T13:43:58Z</td>
<td>2016-05-04T00:00:00Z</td>
<td>-1</td>
<td>No-show</td>
</tr>
<tr>
<th>4</th>
<td>3787481966821</td>
<td>5655637</td>
<td>2016-05-04T06:50:57Z</td>
<td>2016-05-03T00:00:00Z</td>
<td>-1</td>
<td>No-show</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-9d7041a5-a9c0-4301-94ad-715f7d9edaee')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-9d7041a5-a9c0-4301-94ad-715f7d9edaee button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9d7041a5-a9c0-4301-94ad-715f7d9edaee');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Longer-Waiting-Time-Was-Associated-with-Higher-No-Show-Rates">Key Finding: Longer Waiting Time Was Associated with Higher No-Show Rates<a class="anchor-link" href="#Key-Finding:-Longer-Waiting-Time-Was-Associated-with-Higher-No-Show-Rates">¶</a></h3><p>I found that the no-show rate increased substantially as the time between scheduling and the appointment became longer. Same-day appointments had a no-show rate of only 4.65%, compared with 24.15% for appointments scheduled 1–7 days in advance, 31.72% for 8–30 days, and 33.00% for 31+ days.</p>
<p>This pattern suggests that waiting_days may provide useful predictive information for the no-show classification model. I also identified 5 records with negative waiting days, which I treated as potential data-quality issues rather than interpreting their 100% no-show rate as a meaningful pattern.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT</span>
<span class="s2">    MIN(age) AS min_age,</span>
<span class="s2">    MAX(age) AS max_age,</span>
<span class="s2">    SUM(CASE WHEN age &lt; 0 THEN 1 ELSE 0 END) AS negative_age_count</span>
<span class="s2">FROM medical_appointments;</span>
<span class="s2">"""</span>

<span class="n">age_check</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">age_check</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-75f0177e-0cb3-4f54-80f2-9cd98100e419">
<div>
<style scoped="">
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
<th>Minimum Age</th>
<th>Maximum Age</th>
<th>Negative Age Count</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>-1</td>
<td>115</td>
<td>1</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-75f0177e-0cb3-4f54-80f2-9cd98100e419')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-75f0177e-0cb3-4f54-80f2-9cd98100e419 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-75f0177e-0cb3-4f54-80f2-9cd98100e419');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT</span>
<span class="s2">    age,</span>
<span class="s2">    COUNT(*) AS appointment_count,</span>
<span class="s2">    SUM(CASE WHEN no_show = 'Yes' THEN 1 ELSE 0 END) AS no_show_count</span>
<span class="s2">FROM medical_appointments</span>
<span class="s2">WHERE age &lt; 0 OR age &gt; 100</span>
<span class="s2">GROUP BY age</span>
<span class="s2">ORDER BY age;</span>
<span class="s2">"""</span>

<span class="n">age_outliers</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">age_outliers</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-a20251fa-2c9f-4b41-83e8-81b5b65e6606">
<div>
<style scoped="">
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
<th>Age</th>
<th>Appointment Count</th>
<th>No-show Count</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>-1</td>
<td>1</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>102</td>
<td>2</td>
<td>0</td>
</tr>
<tr>
<th>2</th>
<td>115</td>
<td>5</td>
<td>3</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-a20251fa-2c9f-4b41-83e8-81b5b65e6606')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-a20251fa-2c9f-4b41-83e8-81b5b65e6606 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a20251fa-2c9f-4b41-83e8-81b5b65e6606');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Age-Data-Quality-Check">Age Data Quality Check<a class="anchor-link" href="#Age-Data-Quality-Check">¶</a></h3><p>I found one record with an invalid age of -1, which I will exclude during preprocessing. I also found a small number of records for patients aged 102 and 115. Because these values are extreme but not inherently impossible, I retained them rather than removing them without additional evidence.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">query</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">SELECT</span>
<span class="s2">    CASE</span>
<span class="s2">        WHEN age BETWEEN 0 AND 17 THEN '0-17'</span>
<span class="s2">        WHEN age BETWEEN 18 AND 34 THEN '18-34'</span>
<span class="s2">        WHEN age BETWEEN 35 AND 49 THEN '35-49'</span>
<span class="s2">        WHEN age BETWEEN 50 AND 64 THEN '50-64'</span>
<span class="s2">        ELSE '65+'</span>
<span class="s2">    END AS age_group,</span>

<span class="s2">    COUNT(*) AS appointment_count,</span>

<span class="s2">    SUM(</span>
<span class="s2">        CASE WHEN no_show = 'Yes' THEN 1 ELSE 0 END</span>
<span class="s2">    ) AS no_show_count,</span>

<span class="s2">    ROUND(</span>
<span class="s2">        AVG(</span>
<span class="s2">            CASE WHEN no_show = 'Yes' THEN 1.0 ELSE 0 END</span>
<span class="s2">        ) * 100,</span>
<span class="s2">        2</span>
<span class="s2">    ) AS no_show_rate</span>

<span class="s2">FROM medical_appointments</span>

<span class="s2">WHERE age &gt;= 0</span>

<span class="s2">GROUP BY</span>
<span class="s2">    CASE</span>
<span class="s2">        WHEN age BETWEEN 0 AND 17 THEN '0-17'</span>
<span class="s2">        WHEN age BETWEEN 18 AND 34 THEN '18-34'</span>
<span class="s2">        WHEN age BETWEEN 35 AND 49 THEN '35-49'</span>
<span class="s2">        WHEN age BETWEEN 50 AND 64 THEN '50-64'</span>
<span class="s2">        ELSE '65+'</span>
<span class="s2">    END</span>

<span class="s2">ORDER BY</span>
<span class="s2">    MIN(age);</span>
<span class="s2">"""</span>

<span class="n">age_group_result</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_sql</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">engine</span><span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span><span class="n">age_group_result</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-b41c1725-2fa0-4a76-a30c-d499393f568a">
<div>
<style scoped="">
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
<th>Age Group</th>
<th>Appointment Count</th>
<th>No-show Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>0-17</td>
<td>27379</td>
<td>5997</td>
<td>21.90</td>
</tr>
<tr>
<th>1</th>
<td>18-34</td>
<td>24246</td>
<td>5814</td>
<td>23.98</td>
</tr>
<tr>
<th>2</th>
<td>35-49</td>
<td>21865</td>
<td>4490</td>
<td>20.54</td>
</tr>
<tr>
<th>3</th>
<td>50-64</td>
<td>22634</td>
<td>3785</td>
<td>16.72</td>
</tr>
<tr>
<th>4</th>
<td>65+</td>
<td>14402</td>
<td>2233</td>
<td>15.50</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-b41c1725-2fa0-4a76-a30c-d499393f568a')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-b41c1725-2fa0-4a76-a30c-d499393f568a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b41c1725-2fa0-4a76-a30c-d499393f568a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-No-Show-Rates-Varied-Across-Age-Groups">Key Finding: No-Show Rates Varied Across Age Groups<a class="anchor-link" href="#Key-Finding:-No-Show-Rates-Varied-Across-Age-Groups">¶</a></h3><p>I found that the 18–34 age group had the highest no-show rate at 23.98%, followed by patients aged 0–17 at 21.90% and 35–49 at 20.54%.</p>
<p>No-show rates were lower among older patients, decreasing to 16.72% for ages 50–64 and 15.50% for ages 65+. This pattern suggests that age may provide useful information for predicting no-show behavior, so I will retain age as a candidate feature for the machine learning models.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="4.-Exploratory-Data-Analysis-(EDA)-in-Python">4. Exploratory Data Analysis (EDA) in Python<a class="anchor-link" href="#4.-Exploratory-Data-Analysis-(EDA)-in-Python">¶</a></h2><p>After the initial SQL exploration, I used the dataset retrieved from Neon PostgreSQL to check data quality and examine the main variables that may be useful for predicting appointment no-shows.</p>
<p>I focused on:</p>
<ol>
<li>Checking missing values and duplicate records,</li>
<li>Reviewing invalid or unusual values,</li>
<li>Visualizing the target distribution and selected predictors, and</li>
<li>Preparing the data for feature engineering and machine learning.</li>
</ol>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Data-Quality-Check">Data Quality Check<a class="anchor-link" href="#Data-Quality-Check">¶</a></h3><p>Before analyzing no-show patterns, I checked the dataset for missing values, duplicate records, and unusual values that could affect the analysis or machine learning models.</p>
<p>I used a copy of the data retrieved from Neon PostgreSQL so that the original df_sql DataFrame remained unchanged during the EDA process.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">eda_df</span> <span class="o">=</span> <span class="n">df_sql</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">data_quality_summary</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Metric"</span><span class="p">:</span> <span class="p">[</span>
        <span class="s2">"Total Records"</span><span class="p">,</span>
        <span class="s2">"Total Variables"</span><span class="p">,</span>
        <span class="s2">"Missing Values"</span><span class="p">,</span>
        <span class="s2">"Duplicate Records"</span>
    <span class="p">],</span>
    <span class="s2">"Result"</span><span class="p">:</span> <span class="p">[</span>
        <span class="n">eda_df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span>
        <span class="n">eda_df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span>
        <span class="n">eda_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">(),</span>
        <span class="n">eda_df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
    <span class="p">]</span>
<span class="p">})</span>

<span class="n">data_quality_summary</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-5b7c3ffc-f3c8-47f1-82fc-9c4a1096312d">
<div>
<style scoped="">
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
<th>Metric</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Total Records</td>
<td>110527</td>
</tr>
<tr>
<th>1</th>
<td>Total Variables</td>
<td>14</td>
</tr>
<tr>
<th>2</th>
<td>Missing Values</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>Duplicate Records</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-5b7c3ffc-f3c8-47f1-82fc-9c4a1096312d')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-5b7c3ffc-f3c8-47f1-82fc-9c4a1096312d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5b7c3ffc-f3c8-47f1-82fc-9c4a1096312d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="id_2c0bf6fe-7cc2-4d95-b538-44753d7901d2">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('data_quality_summary')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_2c0bf6fe-7cc2-4d95-b538-44753d7901d2 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('data_quality_summary');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-No-Missing-or-Duplicate-Records">Key Finding: No Missing or Duplicate Records<a class="anchor-link" href="#Key-Finding:-No-Missing-or-Duplicate-Records">¶</a></h3><p>I found no missing values or duplicate records in the dataset, so no imputation or duplicate removal was required.</p>
<p>However, the earlier SQL checks identified one invalid age value of -1 and five records with negative waiting days. I will address these records during preprocessing before building the machine learning models.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span> <span class="mi">4</span><span class="p">))</span>

<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span>
    <span class="n">data</span><span class="o">=</span><span class="n">eda_df</span><span class="p">,</span>
    <span class="n">x</span><span class="o">=</span><span class="s2">"no_show"</span><span class="p">,</span>
    <span class="n">order</span><span class="o">=</span><span class="p">[</span><span class="s2">"No"</span><span class="p">,</span> <span class="s2">"Yes"</span><span class="p">]</span>
<span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Appointment Outcome Distribution"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Appointment Outcome"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Number of Appointments"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="s2">"Show"</span><span class="p">,</span> <span class="s2">"No-show"</span><span class="p">])</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img alt="No description has been provided for this image" src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-No-Show-Is-the-Minority-Class">Key Finding: No-Show Is the Minority Class<a class="anchor-link" href="#Key-Finding:-No-Show-Is-the-Minority-Class">¶</a></h3><p>I observed a clear class imbalance, with 20.19% of appointments resulting in a no-show and 79.81% resulting in attendance.</p>
<p>Since the goal of this project is to predict no-shows, I will account for this imbalance during model development by using stratified sampling and evaluating no-show recall alongside precision, F1-score, and ROC-AUC.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="5.-Feature-Engineering">5. Feature Engineering<a class="anchor-link" href="#5.-Feature-Engineering">¶</a></h2><p>To prepare the data for machine learning, I created several features from the original appointment information.</p>
<p>I converted the appointment outcome into a binary target, calculated the number of days between scheduling and the appointment as waiting_days, and extracted weekday information from the date variables. These features were designed using information that would already be available before the appointment outcome was known.</p>
<p>I also removed records with clearly invalid values identified during the earlier data-quality checks before building the prediction models.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span> <span class="o">=</span> <span class="n">eda_df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span><span class="p">[</span><span class="s2">"scheduled_day"</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"scheduled_day"</span><span class="p">],</span>
    <span class="n">utc</span><span class="o">=</span><span class="kc">True</span>
<span class="p">)</span>

<span class="n">model_df</span><span class="p">[</span><span class="s2">"appointment_day"</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"appointment_day"</span><span class="p">],</span>
    <span class="n">utc</span><span class="o">=</span><span class="kc">True</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">display_table</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[[</span><span class="s2">"scheduled_day"</span><span class="p">,</span> <span class="s2">"appointment_day"</span><span class="p">]]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-f58411b9-dd15-450b-a801-09b519c6521e">
<div>
<style scoped="">
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
<th>Scheduled Day</th>
<th>Appointment Day</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>2016-04-29 18:38:08+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
</tr>
<tr>
<th>1</th>
<td>2016-04-29 16:08:27+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
</tr>
<tr>
<th>2</th>
<td>2016-04-29 16:19:04+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
</tr>
<tr>
<th>3</th>
<td>2016-04-29 17:29:31+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
</tr>
<tr>
<th>4</th>
<td>2016-04-29 16:07:23+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-f58411b9-dd15-450b-a801-09b519c6521e')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-f58411b9-dd15-450b-a801-09b519c6521e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f58411b9-dd15-450b-a801-09b519c6521e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span><span class="p">[</span><span class="s2">"no_show_binary"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"no_show"</span><span class="p">]</span>
    <span class="o">.</span><span class="n">map</span><span class="p">({</span><span class="s2">"No"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span> <span class="s2">"Yes"</span><span class="p">:</span> <span class="mi">1</span><span class="p">})</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">display_table</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span>
        <span class="p">[</span><span class="s2">"no_show"</span><span class="p">,</span> <span class="s2">"no_show_binary"</span><span class="p">]</span>
    <span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-96972e7a-929f-4810-8689-b33ca1f3c240">
<div>
<style scoped="">
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
<th>Appointment Outcome</th>
<th>No-show Binary</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>2</th>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>Show</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-96972e7a-929f-4810-8689-b33ca1f3c240')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-96972e7a-929f-4810-8689-b33ca1f3c240 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-96972e7a-929f-4810-8689-b33ca1f3c240');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span><span class="p">[</span><span class="s2">"waiting_days"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"appointment_day"</span><span class="p">]</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">normalize</span><span class="p">()</span>
    <span class="o">-</span> <span class="n">model_df</span><span class="p">[</span><span class="s2">"scheduled_day"</span><span class="p">]</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">normalize</span><span class="p">()</span>
<span class="p">)</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">days</span>

<span class="n">display_table</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span>
        <span class="p">[</span><span class="s2">"scheduled_day"</span><span class="p">,</span> <span class="s2">"appointment_day"</span><span class="p">,</span> <span class="s2">"waiting_days"</span><span class="p">]</span>
    <span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-be7b1003-7f61-481e-834e-31d8aa5e2b15">
<div>
<style scoped="">
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
<th>Scheduled Day</th>
<th>Appointment Day</th>
<th>Waiting Days</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>2016-04-29 18:38:08+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>2016-04-29 16:08:27+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
</tr>
<tr>
<th>2</th>
<td>2016-04-29 16:19:04+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>2016-04-29 17:29:31+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>2016-04-29 16:07:23+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-be7b1003-7f61-481e-834e-31d8aa5e2b15')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-be7b1003-7f61-481e-834e-31d8aa5e2b15 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-be7b1003-7f61-481e-834e-31d8aa5e2b15');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span><span class="p">[</span><span class="s2">"scheduled_weekday"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"scheduled_day"</span><span class="p">]</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">day_name</span><span class="p">()</span>
<span class="p">)</span>

<span class="n">model_df</span><span class="p">[</span><span class="s2">"appointment_weekday"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"appointment_day"</span><span class="p">]</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">day_name</span><span class="p">()</span>
<span class="p">)</span>

<span class="n">display_table</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span>
        <span class="p">[</span>
            <span class="s2">"scheduled_day"</span><span class="p">,</span>
            <span class="s2">"scheduled_weekday"</span><span class="p">,</span>
            <span class="s2">"appointment_day"</span><span class="p">,</span>
            <span class="s2">"appointment_weekday"</span>
        <span class="p">]</span>
    <span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-b2dc1edd-4270-46cd-b485-038c1089a524">
<div>
<style scoped="">
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
<th>Scheduled Day</th>
<th>Scheduled Weekday</th>
<th>Appointment Day</th>
<th>Appointment Weekday</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>2016-04-29 18:38:08+00:00</td>
<td>Friday</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>Friday</td>
</tr>
<tr>
<th>1</th>
<td>2016-04-29 16:08:27+00:00</td>
<td>Friday</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>Friday</td>
</tr>
<tr>
<th>2</th>
<td>2016-04-29 16:19:04+00:00</td>
<td>Friday</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>Friday</td>
</tr>
<tr>
<th>3</th>
<td>2016-04-29 17:29:31+00:00</td>
<td>Friday</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>Friday</td>
</tr>
<tr>
<th>4</th>
<td>2016-04-29 16:07:23+00:00</td>
<td>Friday</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>Friday</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-b2dc1edd-4270-46cd-b485-038c1089a524')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-b2dc1edd-4270-46cd-b485-038c1089a524 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b2dc1edd-4270-46cd-b485-038c1089a524');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span> <span class="o">=</span> <span class="n">model_df</span><span class="p">[</span>
    <span class="p">(</span><span class="n">model_df</span><span class="p">[</span><span class="s2">"age"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">0</span><span class="p">)</span>
    <span class="o">&amp;</span> <span class="p">(</span><span class="n">model_df</span><span class="p">[</span><span class="s2">"waiting_days"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">0</span><span class="p">)</span>
<span class="p">]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>(110521, 18)</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">display_table</span><span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span>
        <span class="p">[</span>
            <span class="s2">"age"</span><span class="p">,</span>
            <span class="s2">"scheduled_day"</span><span class="p">,</span>
            <span class="s2">"appointment_day"</span><span class="p">,</span>
            <span class="s2">"waiting_days"</span><span class="p">,</span>
            <span class="s2">"scheduled_weekday"</span><span class="p">,</span>
            <span class="s2">"appointment_weekday"</span><span class="p">,</span>
            <span class="s2">"no_show"</span><span class="p">,</span>
            <span class="s2">"no_show_binary"</span>
        <span class="p">]</span>
    <span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-6d0daae1-7d28-4ca8-a63f-9a5ab46fbfef">
<div>
<style scoped="">
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
<th>Age</th>
<th>Scheduled Day</th>
<th>Appointment Day</th>
<th>Waiting Days</th>
<th>Scheduled Weekday</th>
<th>Appointment Weekday</th>
<th>Appointment Outcome</th>
<th>No-show Binary</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>62</td>
<td>2016-04-29 18:38:08+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
<td>Friday</td>
<td>Friday</td>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>56</td>
<td>2016-04-29 16:08:27+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
<td>Friday</td>
<td>Friday</td>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>2</th>
<td>62</td>
<td>2016-04-29 16:19:04+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
<td>Friday</td>
<td>Friday</td>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>8</td>
<td>2016-04-29 17:29:31+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
<td>Friday</td>
<td>Friday</td>
<td>Show</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>56</td>
<td>2016-04-29 16:07:23+00:00</td>
<td>2016-04-29 00:00:00+00:00</td>
<td>0</td>
<td>Friday</td>
<td>Friday</td>
<td>Show</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-6d0daae1-7d28-4ca8-a63f-9a5ab46fbfef')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-6d0daae1-7d28-4ca8-a63f-9a5ab46fbfef button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6d0daae1-7d28-4ca8-a63f-9a5ab46fbfef');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Modeling-Dataset-Prepared">Key Finding: Modeling Dataset Prepared<a class="anchor-link" href="#Key-Finding:-Modeling-Dataset-Prepared">¶</a></h3><p>I created a binary no-show target and derived waiting_days, scheduled_weekday, and appointment_weekday from the original appointment information.</p>
<p>After removing 1 record with an invalid age and 5 records with negative waiting days, the modeling dataset contained 110,521 appointments. These engineered features will be evaluated as candidate predictors in the machine learning models.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="6.-Feature-Analysis">6. Feature Analysis<a class="anchor-link" href="#6.-Feature-Analysis">¶</a></h2><p>Before building the machine learning models, I examined which patient and appointment characteristics showed meaningful relationships with the no-show outcome.</p>
<p>Because the dataset contains both numerical features and categorical features, I analyzed them differently. For numerical variables such as age and waiting_days, I compared their distributions and relationships with the binary no-show target. For categorical variables such as gender, sms_received, and appointment-related categories, I compared no-show rates across groups.</p>
<p>The goal of this section is not to treat these relationships as causal effects, but to identify which variables may provide useful predictive information before model development.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">model_df</span><span class="p">[</span><span class="s2">"appointment_outcome"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">model_df</span><span class="p">[</span><span class="s2">"no_show"</span><span class="p">]</span>
    <span class="o">.</span><span class="n">map</span><span class="p">({</span><span class="s2">"No"</span><span class="p">:</span> <span class="s2">"Show"</span><span class="p">,</span> <span class="s2">"Yes"</span><span class="p">:</span> <span class="s2">"No-show"</span><span class="p">})</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">numerical_features</span> <span class="o">=</span> <span class="p">[</span><span class="s2">"age"</span><span class="p">,</span> <span class="s2">"waiting_days"</span><span class="p">]</span>

<span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">numerical_features</span><span class="p">:</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span> <span class="mi">4</span><span class="p">))</span>

    <span class="n">sns</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span>
        <span class="n">data</span><span class="o">=</span><span class="n">model_df</span><span class="p">,</span>
        <span class="n">x</span><span class="o">=</span><span class="s2">"appointment_outcome"</span><span class="p">,</span>
        <span class="n">y</span><span class="o">=</span><span class="n">feature</span><span class="p">,</span>
        <span class="n">order</span><span class="o">=</span><span class="p">[</span><span class="s2">"Show"</span><span class="p">,</span> <span class="s2">"No-show"</span><span class="p">]</span>
    <span class="p">)</span>

    <span class="n">display_name</span> <span class="o">=</span> <span class="n">feature</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">"_"</span><span class="p">,</span> <span class="s2">" "</span><span class="p">)</span><span class="o">.</span><span class="n">title</span><span class="p">()</span>

    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">display_name</span><span class="si">}</span><span class="s2"> by Appointment Outcome"</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Appointment Outcome"</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="n">display_name</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img alt="No description has been provided for this image" src="data:image/png;base64,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YuHChfDx8UGzZs0wfPhwuLm54erVq/jll19w+vRpAMDMmTOxe/du+Pj44F//+heMjIywfPlypKenY86cOUV/wgugxutA5YB+Do4hKp4ePXqIqampPHjwoMA+QUFBYmxsLLdv3xaRJ4cKDhgwQDnxV1BQkERERAgAWb9+vc59L1++LIMGDRIHBwcxNjaWV155Rbp37y4///zzP9b19Em25s2bJ87OzqLVaqVt27bKeRCelZCQIIaGhlKnTh1Vtr0oNeV34i97e3uZNm2azuG7IiKpqakybtw4cXJyEmNjY6ldu3aRTvx1/PhxnX779u3Lc1hpYmKidOvWTaysrPI98dezY0ybNk0AKIfvPrtdz1qxYoU0b95czMzMxMrKSjw8PGTChAkSHx+vU3+3bt3y3De/Q1RXrlwpNWvWFENDw0IfInvw4EHx9/eXqlWrirGxsbi4uMiwYcPk6tWr+fb/4YcfpGbNmmJiYiJNmjSR//73v3kOixURiYyMlObNm4uJiUmeQ2TPnj0rb775plSqVElMTU2lbt26OudZEXlyiLafn59YWlqKubm5dOzYUSIjI3X6lObrQBWHRqSYez8RlWPh4eF48803cfjwYXh7e+ulhtu3b8PR0RGff/45pk6dWuLjX716FW5ubvj666+VmZSCBAUF4eeff0ZaWlqJ10FEBHAfDnoJPPs9eXZ2NhYtWgRra2s0a9ZMT1UBq1evRnZ2Nt555x291UBEVFq4DwdVeGPGjMGjR4/g5eWF9PR0bN68GZGRkZg1a5Ze9pDfu3cvzp07h+DgYAQEBOQ5WoSIqCJi4KAKr1OnTpg3bx527NiBx48fo1atWli0aJHOWTVL04wZMxAZGQlvb28sWrRILzUQEZU27sNBREREquM+HERERKQ6vQaOgwcPokePHnBycoJGo0F4eLiyLjMzE59++ik8PDxgYWEBJycn5aJXT7t79y4GDhwIa2trVKpUCUOHDuWe9kRERGWMXvfhePDgARo3bowhQ4agV69eOusePnyIU6dOYerUqWjcuDHu3buHDz/8ED179sSJEyeUfgMHDkRCQgJ2796NzMxMvPvuuxg+fDh+/PHHQteRk5OD+Ph4WFlZqXZKZSIioopIRJCamgonJycYGPzDPIZezwLyFACyZcuWf+xz7NgxASDXrl0TEZFz587lOfnMzp07RaPRyN9/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
"/>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img alt="No description has been provided for this image" src="data:image/png;base64,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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">numerical_summary</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">model_df</span>
    <span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s2">"appointment_outcome"</span><span class="p">)[[</span><span class="s2">"age"</span><span class="p">,</span> <span class="s2">"waiting_days"</span><span class="p">]]</span>
    <span class="o">.</span><span class="n">agg</span><span class="p">([</span><span class="s2">"mean"</span><span class="p">,</span> <span class="s2">"median"</span><span class="p">])</span>
    <span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
<span class="p">)</span>

<span class="n">numerical_summary_display</span> <span class="o">=</span> <span class="n">numerical_summary</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>

<span class="n">numerical_summary_display</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span>
    <span class="s2">"Appointment Outcome"</span><span class="p">,</span>
    <span class="s2">"Age Mean"</span><span class="p">,</span>
    <span class="s2">"Age Median"</span><span class="p">,</span>
    <span class="s2">"Waiting Days Mean"</span><span class="p">,</span>
    <span class="s2">"Waiting Days Median"</span>
<span class="p">]</span>

<span class="n">numerical_summary_display</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-fb81a535-71ab-464a-9910-f3d64629b893">
<div>
<style scoped="">
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
<th>Appointment Outcome</th>
<th>Age Mean</th>
<th>Age Median</th>
<th>Waiting Days Mean</th>
<th>Waiting Days Median</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>No-show</td>
<td>34.32</td>
<td>33.0</td>
<td>15.84</td>
<td>11.0</td>
</tr>
<tr>
<th>1</th>
<td>Show</td>
<td>37.79</td>
<td>38.0</td>
<td>8.75</td>
<td>2.0</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-fb81a535-71ab-464a-9910-f3d64629b893')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-fb81a535-71ab-464a-9910-f3d64629b893 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fb81a535-71ab-464a-9910-f3d64629b893');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="id_0ec9d315-30b6-4d91-a951-52b0001044a8">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('numerical_summary_display')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0ec9d315-30b6-4d91-a951-52b0001044a8 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('numerical_summary_display');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Waiting-Days-Showed-a-Clearer-Difference-Than-Age">Key Finding: Waiting Days Showed a Clearer Difference Than Age<a class="anchor-link" href="#Key-Finding:-Waiting-Days-Showed-a-Clearer-Difference-Than-Age">¶</a></h3><p>I found that patients who missed their appointments were somewhat younger on average than those who attended. The mean age was 34.32 years for no-shows compared with 37.79 years for shows, although the two age distributions overlapped substantially.</p>
<p>The difference was more pronounced for waiting_days. No-show appointments had an average of 15.84 waiting days and a median of 11 days, compared with an average of 8.75 waiting days and a median of 2 days for attended appointments.</p>
<p>Based on these patterns, I expect waiting_days to provide stronger predictive information than age, although both variables will be retained as candidate features for the machine learning models.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">categorical_features</span> <span class="o">=</span> <span class="p">[</span>
    <span class="s2">"gender"</span><span class="p">,</span>
    <span class="s2">"scholarship"</span><span class="p">,</span>
    <span class="s2">"appointment_weekday"</span>
<span class="p">]</span>

<span class="n">feature_labels</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"gender"</span><span class="p">:</span> <span class="s2">"Gender"</span><span class="p">,</span>
    <span class="s2">"scholarship"</span><span class="p">:</span> <span class="s2">"Scholarship"</span><span class="p">,</span>
    <span class="s2">"appointment_weekday"</span><span class="p">:</span> <span class="s2">"Appointment Weekday"</span>
<span class="p">}</span>

<span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">categorical_features</span><span class="p">:</span>
    <span class="n">summary</span> <span class="o">=</span> <span class="p">(</span>
        <span class="n">model_df</span>
        <span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">feature</span><span class="p">)[</span><span class="s2">"no_show_binary"</span><span class="p">]</span>
        <span class="o">.</span><span class="n">agg</span><span class="p">([</span><span class="s2">"count"</span><span class="p">,</span> <span class="s2">"mean"</span><span class="p">])</span>
        <span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
    <span class="p">)</span>

    <span class="n">summary</span><span class="p">[</span><span class="s2">"no_show_rate"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="n">summary</span><span class="p">[</span><span class="s2">"mean"</span><span class="p">]</span> <span class="o">*</span> <span class="mi">100</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">feature</span> <span class="o">==</span> <span class="s2">"appointment_weekday"</span><span class="p">:</span>
        <span class="n">weekday_order</span> <span class="o">=</span> <span class="p">[</span>
            <span class="s2">"Monday"</span><span class="p">,</span>
            <span class="s2">"Tuesday"</span><span class="p">,</span>
            <span class="s2">"Wednesday"</span><span class="p">,</span>
            <span class="s2">"Thursday"</span><span class="p">,</span>
            <span class="s2">"Friday"</span><span class="p">,</span>
            <span class="s2">"Saturday"</span><span class="p">,</span>
            <span class="s2">"Sunday"</span>
        <span class="p">]</span>

        <span class="n">summary</span><span class="p">[</span><span class="n">feature</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Categorical</span><span class="p">(</span>
            <span class="n">summary</span><span class="p">[</span><span class="n">feature</span><span class="p">],</span>
            <span class="n">categories</span><span class="o">=</span><span class="n">weekday_order</span><span class="p">,</span>
            <span class="n">ordered</span><span class="o">=</span><span class="kc">True</span>
        <span class="p">)</span>

        <span class="n">summary</span> <span class="o">=</span> <span class="n">summary</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">feature</span><span class="p">)</span>

    <span class="n">summary_display</span> <span class="o">=</span> <span class="n">summary</span><span class="p">[</span>
        <span class="p">[</span><span class="n">feature</span><span class="p">,</span> <span class="s2">"count"</span><span class="p">,</span> <span class="s2">"no_show_rate"</span><span class="p">]</span>
    <span class="p">]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

    <span class="n">summary_display</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">feature_labels</span><span class="p">[</span><span class="n">feature</span><span class="p">],</span>
        <span class="s2">"Appointment Count"</span><span class="p">,</span>
        <span class="s2">"No-show Rate (%)"</span>
    <span class="p">]</span>

    <span class="k">if</span> <span class="n">feature</span> <span class="o">==</span> <span class="s2">"scholarship"</span><span class="p">:</span>
        <span class="n">summary_display</span><span class="p">[</span><span class="s2">"Scholarship"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
            <span class="n">summary_display</span><span class="p">[</span><span class="s2">"Scholarship"</span><span class="p">]</span>
            <span class="o">.</span><span class="n">replace</span><span class="p">({</span>
                <span class="mi">0</span><span class="p">:</span> <span class="s2">"No"</span><span class="p">,</span>
                <span class="mi">1</span><span class="p">:</span> <span class="s2">"Yes"</span>
            <span class="p">})</span>
        <span class="p">)</span>

    <span class="n">display</span><span class="p">(</span><span class="n">Markdown</span><span class="p">(</span><span class="sa">f</span><span class="s2">"#### </span><span class="si">{</span><span class="n">feature_labels</span><span class="p">[</span><span class="n">feature</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">))</span>
    <span class="n">display</span><span class="p">(</span><span class="n">summary_display</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_markdown rendered_html output_subarea">
<h4 id="Gender">Gender<a class="anchor-link" href="#Gender">¶</a></h4>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_html rendered_html output_subarea">
<div class="colab-df-container" id="df-abe44ef3-fbf9-4f8d-81bc-ce3488f7b186">
<div>
<style scoped="">
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
<th>Gender</th>
<th>Appointment Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>F</td>
<td>71836</td>
<td>20.31</td>
</tr>
<tr>
<th>1</th>
<td>M</td>
<td>38685</td>
<td>19.96</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-abe44ef3-fbf9-4f8d-81bc-ce3488f7b186')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-abe44ef3-fbf9-4f8d-81bc-ce3488f7b186 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-abe44ef3-fbf9-4f8d-81bc-ce3488f7b186');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="id_6072f3f4-c6cb-4e55-992b-9a7820249ae6">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('summary_display')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_6072f3f4-c6cb-4e55-992b-9a7820249ae6 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_display');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_markdown rendered_html output_subarea">
<h4 id="Scholarship">Scholarship<a class="anchor-link" href="#Scholarship">¶</a></h4>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_html rendered_html output_subarea">
<div class="colab-df-container" id="df-81351887-2452-4d7a-922e-f8cb306347a4">
<div>
<style scoped="">
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
<th>Scholarship</th>
<th>Appointment Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>No</td>
<td>99660</td>
<td>19.80</td>
</tr>
<tr>
<th>1</th>
<td>Yes</td>
<td>10861</td>
<td>23.74</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-81351887-2452-4d7a-922e-f8cb306347a4')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-81351887-2452-4d7a-922e-f8cb306347a4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-81351887-2452-4d7a-922e-f8cb306347a4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="id_603fa44c-dd7c-49c0-ad27-2854b1bf2e1f">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('summary_display')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_603fa44c-dd7c-49c0-ad27-2854b1bf2e1f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_display');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_markdown rendered_html output_subarea">
<h4 id="Appointment-Weekday">Appointment Weekday<a class="anchor-link" href="#Appointment-Weekday">¶</a></h4>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_html rendered_html output_subarea">
<div class="colab-df-container" id="df-7d1247a2-f16d-4170-aca5-2cfa2611c179">
<div>
<style scoped="">
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
<th>Appointment Weekday</th>
<th>Appointment Count</th>
<th>No-show Rate (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>1</th>
<td>Monday</td>
<td>22713</td>
<td>20.64</td>
</tr>
<tr>
<th>4</th>
<td>Tuesday</td>
<td>25638</td>
<td>20.09</td>
</tr>
<tr>
<th>5</th>
<td>Wednesday</td>
<td>25866</td>
<td>19.69</td>
</tr>
<tr>
<th>3</th>
<td>Thursday</td>
<td>17246</td>
<td>19.35</td>
</tr>
<tr>
<th>0</th>
<td>Friday</td>
<td>19019</td>
<td>21.23</td>
</tr>
<tr>
<th>2</th>
<td>Saturday</td>
<td>39</td>
<td>23.08</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-7d1247a2-f16d-4170-aca5-2cfa2611c179')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-7d1247a2-f16d-4170-aca5-2cfa2611c179 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7d1247a2-f16d-4170-aca5-2cfa2611c179');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="id_d3b31b94-f4bc-4820-b255-e39b2037c78f">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('summary_display')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d3b31b94-f4bc-4820-b255-e39b2037c78f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_display');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Categorical-Features-Showed-Modest-Differences-in-No-Show-Rates">Key Finding: Categorical Features Showed Modest Differences in No-Show Rates<a class="anchor-link" href="#Key-Finding:-Categorical-Features-Showed-Modest-Differences-in-No-Show-Rates">¶</a></h3><p>I found little difference in no-show rates by gender, with 20.31% for female patients and 19.96% for male patients.</p>
<p>A larger difference appeared for scholarship status. Patients enrolled in the scholarship program had a no-show rate of 23.74%, compared with 19.80% among patients who were not enrolled.</p>
<p>No-show rates also varied somewhat across appointment weekdays, although most weekdays remained within a relatively narrow range. Saturday had a higher no-show rate of 23.08%, but only 39 appointments occurred on Saturday, so I interpreted this result cautiously because of the very small sample size.</p>
<p>Overall, these categorical variables showed weaker differences than waiting_days, but I will retain them as candidate predictors and evaluate their contribution together in the machine learning models.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Features-Selected-for-Modeling">Features Selected for Modeling<a class="anchor-link" href="#Features-Selected-for-Modeling">¶</a></h3><p>Based on the data-quality checks, feature engineering, and exploratory analysis, I selected patient characteristics and appointment information that would be available before the appointment outcome was known.</p>
<p>I excluded <code>patient_id</code> and <code>appointment_id</code> because they are identifiers rather than meaningful predictive features. I also excluded the original date columns and the original <code>no_show</code> column after deriving the modeling features and binary target from them.</p>
<p>The modeling features include both <strong>numerical</strong> and <strong>categorical</strong> variables, which will require different preprocessing steps before model training.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">numerical_features</span> <span class="o">=</span> <span class="p">[</span>
    <span class="s2">"age"</span><span class="p">,</span>
    <span class="s2">"waiting_days"</span>
<span class="p">]</span>

<span class="n">categorical_features</span> <span class="o">=</span> <span class="p">[</span>
    <span class="s2">"gender"</span><span class="p">,</span>
    <span class="s2">"neighbourhood"</span><span class="p">,</span>
    <span class="s2">"scholarship"</span><span class="p">,</span>
    <span class="s2">"hypertension"</span><span class="p">,</span>
    <span class="s2">"diabetes"</span><span class="p">,</span>
    <span class="s2">"alcoholism"</span><span class="p">,</span>
    <span class="s2">"handicap"</span><span class="p">,</span>
    <span class="s2">"sms_received"</span><span class="p">,</span>
    <span class="s2">"scheduled_weekday"</span><span class="p">,</span>
    <span class="s2">"appointment_weekday"</span>
<span class="p">]</span>

<span class="n">X</span> <span class="o">=</span> <span class="n">model_df</span><span class="p">[</span>
    <span class="n">numerical_features</span> <span class="o">+</span> <span class="n">categorical_features</span>
<span class="p">]</span>

<span class="n">y</span> <span class="o">=</span> <span class="n">model_df</span><span class="p">[</span><span class="s2">"no_show_binary"</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">y</span><span class="o">.</span><span class="n">shape</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>((110521, 12), (110521,))</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="7.-Machine-Learning-Model-Development">7. Machine Learning Model Development<a class="anchor-link" href="#7.-Machine-Learning-Model-Development">¶</a></h2><p>I developed classification models to predict whether an appointment would result in a <strong>no-show (<code>1</code>)</strong> or <strong>attendance (<code>0</code>)</strong>.</p>
<p>Because no-shows account for only about <strong>20% of the observations</strong>, I used stratified sampling to preserve the class distribution when splitting the data. I also planned to evaluate the models using <strong>no-show recall, precision, F1-score, and ROC-AUC</strong>, rather than relying on accuracy alone.</p>
<p>The dataset contains both numerical and categorical predictors, so I applied separate preprocessing steps before model training. Numerical features were standardized where required, while categorical features were converted into machine-readable variables using <strong>one-hot encoding</strong>.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span>
    <span class="n">X</span><span class="p">,</span>
    <span class="n">y</span><span class="p">,</span>
    <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>
    <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
    <span class="n">stratify</span><span class="o">=</span><span class="n">y</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">split_summary</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Dataset"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"Training"</span><span class="p">,</span> <span class="s2">"Test"</span><span class="p">],</span>
    <span class="s2">"Rows"</span><span class="p">:</span> <span class="p">[</span><span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]],</span>
    <span class="s2">"Features"</span><span class="p">:</span> <span class="p">[</span><span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]],</span>
    <span class="s2">"Show (%)"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">(</span><span class="n">y_train</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="o">*</span> <span class="mi">100</span><span class="p">,</span>
        <span class="p">(</span><span class="n">y_test</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="o">*</span> <span class="mi">100</span>
    <span class="p">],</span>
    <span class="s2">"No-show (%)"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">(</span><span class="n">y_train</span> <span class="o">==</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="o">*</span> <span class="mi">100</span><span class="p">,</span>
        <span class="p">(</span><span class="n">y_test</span> <span class="o">==</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="o">*</span> <span class="mi">100</span>
    <span class="p">]</span>
<span class="p">})</span>

<span class="n">split_summary</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-3a3fb910-c7fc-4ecc-b722-7531de488cdc">
<div>
<style scoped="">
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
<th>Dataset</th>
<th>Rows</th>
<th>Features</th>
<th>Show (%)</th>
<th>No-show (%)</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Training</td>
<td>88416</td>
<td>12</td>
<td>79.81</td>
<td>20.19</td>
</tr>
<tr>
<th>1</th>
<td>Test</td>
<td>22105</td>
<td>12</td>
<td>79.81</td>
<td>20.19</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-3a3fb910-c7fc-4ecc-b722-7531de488cdc')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-3a3fb910-c7fc-4ecc-b722-7531de488cdc button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3a3fb910-c7fc-4ecc-b722-7531de488cdc');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Split-Result">Split Result<a class="anchor-link" href="#Split-Result">¶</a></h3><p>I confirmed that the training set contained <strong>88,416 records</strong> and the holdout test set contained <strong>22,105 records</strong>, with <strong>12 predictor variables</strong> in each set. Both datasets retained the same class distribution of <strong>79.81% show appointments</strong> and <strong>20.19% no-shows</strong>, confirming that the stratified split worked as intended.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Preprocessing">Preprocessing<a class="anchor-link" href="#Preprocessing">¶</a></h3><p>Before training the models, I prepared the numerical and categorical features separately.</p>
<p>I standardized the numerical features (<code>age</code> and <code>waiting_days</code>) and used one-hot encoding for the categorical features so that they could be used by the classification models.</p>
<p>I included these preprocessing steps within the modeling pipeline so that the transformations were learned only from the training data.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">numerical_transformer</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>

<span class="n">categorical_transformer</span> <span class="o">=</span> <span class="n">OneHotEncoder</span><span class="p">(</span>
    <span class="n">handle_unknown</span><span class="o">=</span><span class="s2">"ignore"</span>
<span class="p">)</span>

<span class="n">preprocessor</span> <span class="o">=</span> <span class="n">ColumnTransformer</span><span class="p">(</span>
    <span class="n">transformers</span><span class="o">=</span><span class="p">[</span>
        <span class="p">(</span>
            <span class="s2">"num"</span><span class="p">,</span>
            <span class="n">numerical_transformer</span><span class="p">,</span>
            <span class="n">numerical_features</span>
        <span class="p">),</span>
        <span class="p">(</span>
            <span class="s2">"cat"</span><span class="p">,</span>
            <span class="n">categorical_transformer</span><span class="p">,</span>
            <span class="n">categorical_features</span>
        <span class="p">)</span>
    <span class="p">]</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Models">Models<a class="anchor-link" href="#Models">¶</a></h3><p>I selected two classification models for predicting appointment no-shows:</p>
<ul>
<li><strong>Logistic Regression</strong> as a simple and interpretable baseline model</li>
<li><strong>Random Forest</strong> as a nonlinear model that can capture more complex relationships between the predictors</li>
</ul>
<p>Because no-shows represent the minority class, I used balanced class weights in both models so that the models would place greater importance on identifying no-show appointments.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Model-Comparison">Model Comparison<a class="anchor-link" href="#Model-Comparison">¶</a></h3><p>I compared Logistic Regression and Random Forest using stratified 3-fold cross-validation on the training data.</p>
<p>The models were first evaluated at the default classification threshold of 0.5 using accuracy, precision, recall, F1-score, and ROC-AUC.</p>
<p>Because recall depends on the classification threshold, I also compared the two models at a similar no-show recall level. This allowed me to examine which model achieved better precision when identifying approximately the same proportion of actual no-shows.</p>
<p>The holdout test set remained untouched and was used only for the final evaluation of the selected model.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">models</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"Logistic Regression"</span><span class="p">:</span> <span class="n">Pipeline</span><span class="p">([</span>
        <span class="p">(</span><span class="s2">"preprocessor"</span><span class="p">,</span> <span class="n">preprocessor</span><span class="p">),</span>
        <span class="p">(</span>
            <span class="s2">"model"</span><span class="p">,</span>
            <span class="n">LogisticRegression</span><span class="p">(</span>
                <span class="n">max_iter</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span>
                <span class="n">class_weight</span><span class="o">=</span><span class="s2">"balanced"</span><span class="p">,</span>
                <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span>
            <span class="p">)</span>
        <span class="p">)</span>
    <span class="p">]),</span>

    <span class="s2">"Random Forest"</span><span class="p">:</span> <span class="n">Pipeline</span><span class="p">([</span>
        <span class="p">(</span><span class="s2">"preprocessor"</span><span class="p">,</span> <span class="n">preprocessor</span><span class="p">),</span>
        <span class="p">(</span>
            <span class="s2">"model"</span><span class="p">,</span>
            <span class="n">RandomForestClassifier</span><span class="p">(</span>
                <span class="n">n_estimators</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
                <span class="n">class_weight</span><span class="o">=</span><span class="s2">"balanced"</span><span class="p">,</span>
                <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
                <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span>
            <span class="p">)</span>
        <span class="p">)</span>
    <span class="p">])</span>
<span class="p">}</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">cv</span> <span class="o">=</span> <span class="n">StratifiedKFold</span><span class="p">(</span>
    <span class="n">n_splits</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
    <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span>
<span class="p">)</span>

<span class="n">cv_probabilities</span> <span class="o">=</span> <span class="p">{}</span>

<span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">model</span> <span class="ow">in</span> <span class="n">models</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
    <span class="n">cv_probabilities</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="n">cross_val_predict</span><span class="p">(</span>
        <span class="n">model</span><span class="p">,</span>
        <span class="n">X_train</span><span class="p">,</span>
        <span class="n">y_train</span><span class="p">,</span>
        <span class="n">cv</span><span class="o">=</span><span class="n">cv</span><span class="p">,</span>
        <span class="n">method</span><span class="o">=</span><span class="s2">"predict_proba"</span><span class="p">,</span>
        <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span>
    <span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

<span class="n">model_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">y_prob</span> <span class="ow">in</span> <span class="n">cv_probabilities</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
    <span class="n">y_pred</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_prob</span> <span class="o">&gt;=</span> <span class="mf">0.5</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

    <span class="n">model_results</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
        <span class="s2">"Model"</span><span class="p">:</span> <span class="n">name</span><span class="p">,</span>
        <span class="s2">"Threshold"</span><span class="p">:</span> <span class="mf">0.50</span><span class="p">,</span>
        <span class="s2">"Accuracy"</span><span class="p">:</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"Precision"</span><span class="p">:</span> <span class="n">precision_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"Recall"</span><span class="p">:</span> <span class="n">recall_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"F1"</span><span class="p">:</span> <span class="n">f1_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">),</span>
        <span class="s2">"ROC-AUC"</span><span class="p">:</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">y_prob</span><span class="p">)</span>
    <span class="p">})</span>

<span class="n">model_results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">model_results</span><span class="p">)</span>

<span class="n">model_results_df</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-06d21eef-d480-4b25-b201-d05824e3a283">
<div>
<style scoped="">
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
<th>Model</th>
<th>Threshold</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>ROC-AUC</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Logistic Regression</td>
<td>0.5</td>
<td>0.655</td>
<td>0.310</td>
<td>0.580</td>
<td>0.404</td>
<td>0.666</td>
</tr>
<tr>
<th>1</th>
<td>Random Forest</td>
<td>0.5</td>
<td>0.772</td>
<td>0.377</td>
<td>0.202</td>
<td>0.263</td>
<td>0.712</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-06d21eef-d480-4b25-b201-d05824e3a283')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-06d21eef-d480-4b25-b201-d05824e3a283 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-06d21eef-d480-4b25-b201-d05824e3a283');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Comparison-at-the-Same-Recall-Level">Comparison at the Same Recall Level<a class="anchor-link" href="#Comparison-at-the-Same-Recall-Level">¶</a></h3><p>The default 0.5 threshold resulted in substantially different recall values for the two models. To make the comparison more comparable, I used the Logistic Regression recall at the default threshold as the target recall and adjusted the Random Forest classification threshold to reach approximately the same recall.</p>
<p>I then compared precision at this similar recall level.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="c1"># Use Logistic Regression recall at threshold 0.5 as the target</span>
<span class="n">logistic_prob</span> <span class="o">=</span> <span class="n">cv_probabilities</span><span class="p">[</span><span class="s2">"Logistic Regression"</span><span class="p">]</span>
<span class="n">rf_prob</span> <span class="o">=</span> <span class="n">cv_probabilities</span><span class="p">[</span><span class="s2">"Random Forest"</span><span class="p">]</span>

<span class="n">logistic_pred</span> <span class="o">=</span> <span class="p">(</span><span class="n">logistic_prob</span> <span class="o">&gt;=</span> <span class="mf">0.5</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

<span class="n">target_recall</span> <span class="o">=</span> <span class="n">recall_score</span><span class="p">(</span>
    <span class="n">y_train</span><span class="p">,</span>
    <span class="n">logistic_pred</span>
<span class="p">)</span>

<span class="c1"># Find a Random Forest threshold that produces similar recall</span>
<span class="n">thresholds</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mf">0.05</span><span class="p">,</span> <span class="mf">0.96</span><span class="p">,</span> <span class="mf">0.01</span><span class="p">)</span>

<span class="n">rf_threshold_results</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">threshold</span> <span class="ow">in</span> <span class="n">thresholds</span><span class="p">:</span>
    <span class="n">rf_pred</span> <span class="o">=</span> <span class="p">(</span><span class="n">rf_prob</span> <span class="o">&gt;=</span> <span class="n">threshold</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

    <span class="n">rf_threshold_results</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
        <span class="s2">"Threshold"</span><span class="p">:</span> <span class="n">threshold</span><span class="p">,</span>
        <span class="s2">"Recall"</span><span class="p">:</span> <span class="n">recall_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">rf_pred</span><span class="p">),</span>
        <span class="s2">"Precision"</span><span class="p">:</span> <span class="n">precision_score</span><span class="p">(</span>
            <span class="n">y_train</span><span class="p">,</span>
            <span class="n">rf_pred</span><span class="p">,</span>
            <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span>
        <span class="p">)</span>
    <span class="p">})</span>

<span class="n">rf_threshold_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">rf_threshold_results</span><span class="p">)</span>

<span class="n">rf_threshold_df</span><span class="p">[</span><span class="s2">"Recall Difference"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">rf_threshold_df</span><span class="p">[</span><span class="s2">"Recall"</span><span class="p">]</span> <span class="o">-</span> <span class="n">target_recall</span>
<span class="p">)</span><span class="o">.</span><span class="n">abs</span><span class="p">()</span>

<span class="n">best_rf_threshold</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">rf_threshold_df</span>
    <span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="s2">"Recall Difference"</span><span class="p">)</span>
    <span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="p">)</span>

<span class="n">same_recall_comparison</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Model"</span><span class="p">:</span> <span class="p">[</span>
        <span class="s2">"Logistic Regression"</span><span class="p">,</span>
        <span class="s2">"Random Forest"</span>
    <span class="p">],</span>
    <span class="s2">"Threshold"</span><span class="p">:</span> <span class="p">[</span>
        <span class="mf">0.50</span><span class="p">,</span>
        <span class="n">best_rf_threshold</span><span class="p">[</span><span class="s2">"Threshold"</span><span class="p">]</span>
    <span class="p">],</span>
    <span class="s2">"Recall"</span><span class="p">:</span> <span class="p">[</span>
        <span class="n">target_recall</span><span class="p">,</span>
        <span class="n">best_rf_threshold</span><span class="p">[</span><span class="s2">"Recall"</span><span class="p">]</span>
    <span class="p">],</span>
    <span class="s2">"Precision"</span><span class="p">:</span> <span class="p">[</span>
        <span class="n">precision_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span> <span class="n">logistic_pred</span><span class="p">),</span>
        <span class="n">best_rf_threshold</span><span class="p">[</span><span class="s2">"Precision"</span><span class="p">]</span>
    <span class="p">]</span>
<span class="p">})</span>

<span class="n">same_recall_comparison</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-31f62c51-4cbc-488b-a656-6e24a6acda63">
<div>
<style scoped="">
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
<th>Model</th>
<th>Threshold</th>
<th>Recall</th>
<th>Precision</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Logistic Regression</td>
<td>0.50</td>
<td>0.580</td>
<td>0.310</td>
</tr>
<tr>
<th>1</th>
<td>Random Forest</td>
<td>0.24</td>
<td>0.576</td>
<td>0.327</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-31f62c51-4cbc-488b-a656-6e24a6acda63')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-31f62c51-4cbc-488b-a656-6e24a6acda63 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-31f62c51-4cbc-488b-a656-6e24a6acda63');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Final-Model-Selection">Final Model Selection<a class="anchor-link" href="#Final-Model-Selection">¶</a></h3><p>At the default classification threshold of 0.5, Logistic Regression achieved substantially higher no-show recall (0.580) than Random Forest (0.202). However, Random Forest achieved a higher ROC-AUC (0.712 vs. 0.666), suggesting stronger overall ability to distinguish between show and no-show appointments.</p>
<p>Because recall depends on the classification threshold, I also compared the two models at a similar recall level. Using a threshold of 0.24, Random Forest achieved a recall of 0.576, which was close to Logistic Regression's recall of 0.580 at the default 0.5 threshold.</p>
<p>At this comparable recall level, Random Forest achieved higher precision (0.327 vs. 0.310). This means that when both models identified a similar proportion of actual no-shows, Random Forest produced slightly fewer false-positive predictions.</p>
<p>Based on its higher ROC-AUC and higher precision at a comparable recall level, I selected Random Forest with a classification threshold of 0.24 as the final model for holdout test evaluation.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="8.-Final-Model-Evaluation">8. Final Model Evaluation<a class="anchor-link" href="#8.-Final-Model-Evaluation">¶</a></h2><p>After selecting Random Forest based on the cross-validation results and the same-recall comparison, I retrained the model using the full training dataset and evaluated it on the untouched holdout test set.</p>
<p>I applied the selected classification threshold of 0.24, which produced a recall level comparable to Logistic Regression during cross-validation while achieving slightly higher precision.</p>
<p>I evaluated the final model using accuracy, precision, recall, F1-score, and ROC-AUC, with particular attention to its ability to identify actual no-show appointments.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">final_model</span> <span class="o">=</span> <span class="n">models</span><span class="p">[</span><span class="s2">"Random Forest"</span><span class="p">]</span>
<span class="n">final_threshold</span> <span class="o">=</span> <span class="mf">0.24</span>

<span class="n">final_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

<span class="n">y_test_prob</span> <span class="o">=</span> <span class="n">final_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">y_test_pred</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_test_prob</span> <span class="o">&gt;=</span> <span class="n">final_threshold</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

<span class="n">final_results</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Model"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"Random Forest"</span><span class="p">],</span>
    <span class="s2">"Threshold"</span><span class="p">:</span> <span class="p">[</span><span class="n">final_threshold</span><span class="p">],</span>
    <span class="s2">"Accuracy"</span><span class="p">:</span> <span class="p">[</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred</span><span class="p">)],</span>
    <span class="s2">"Precision"</span><span class="p">:</span> <span class="p">[</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred</span><span class="p">)],</span>
    <span class="s2">"Recall"</span><span class="p">:</span> <span class="p">[</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred</span><span class="p">)],</span>
    <span class="s2">"F1"</span><span class="p">:</span> <span class="p">[</span><span class="n">f1_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred</span><span class="p">)],</span>
    <span class="s2">"ROC-AUC"</span><span class="p">:</span> <span class="p">[</span><span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_prob</span><span class="p">)]</span>
<span class="p">})</span>

<span class="n">final_results</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-ad8591d3-1699-4553-92cf-1be5d3ba25e5">
<div>
<style scoped="">
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
<th>Model</th>
<th>Threshold</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>ROC-AUC</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Random Forest</td>
<td>0.24</td>
<td>0.677</td>
<td>0.33</td>
<td>0.585</td>
<td>0.422</td>
<td>0.718</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-ad8591d3-1699-4553-92cf-1be5d3ba25e5')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-ad8591d3-1699-4553-92cf-1be5d3ba25e5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ad8591d3-1699-4553-92cf-1be5d3ba25e5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Group-Level-Recall-Check">Group-Level Recall Check<a class="anchor-link" href="#Group-Level-Recall-Check">¶</a></h3><p>To examine whether the final model performed differently across patient groups, I compared no-show recall across age groups and scholarship status.</p>
<p>This check helps identify whether the model was substantially less effective at detecting no-shows for particular groups.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">group_results</span> <span class="o">=</span> <span class="n">X_test</span><span class="p">[[</span><span class="s2">"age"</span><span class="p">,</span> <span class="s2">"scholarship"</span><span class="p">]]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

<span class="n">group_results</span><span class="p">[</span><span class="s2">"Actual"</span><span class="p">]</span> <span class="o">=</span> <span class="n">y_test</span><span class="o">.</span><span class="n">values</span>
<span class="n">group_results</span><span class="p">[</span><span class="s2">"Predicted"</span><span class="p">]</span> <span class="o">=</span> <span class="n">y_test_pred</span>

<span class="n">group_results</span><span class="p">[</span><span class="s2">"Age Group"</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">cut</span><span class="p">(</span>
    <span class="n">group_results</span><span class="p">[</span><span class="s2">"age"</span><span class="p">],</span>
    <span class="n">bins</span><span class="o">=</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">17</span><span class="p">,</span> <span class="mi">34</span><span class="p">,</span> <span class="mi">49</span><span class="p">,</span> <span class="mi">64</span><span class="p">,</span> <span class="nb">float</span><span class="p">(</span><span class="s2">"inf"</span><span class="p">)],</span>
    <span class="n">labels</span><span class="o">=</span><span class="p">[</span><span class="s2">"0-17"</span><span class="p">,</span> <span class="s2">"18-34"</span><span class="p">,</span> <span class="s2">"35-49"</span><span class="p">,</span> <span class="s2">"50-64"</span><span class="p">,</span> <span class="s2">"65+"</span><span class="p">]</span>
<span class="p">)</span>

<span class="n">age_recall</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">group_results</span>
    <span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s2">"Age Group"</span><span class="p">,</span> <span class="n">observed</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="o">.</span><span class="n">apply</span><span class="p">(</span>
        <span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">recall_score</span><span class="p">(</span>
            <span class="n">x</span><span class="p">[</span><span class="s2">"Actual"</span><span class="p">],</span>
            <span class="n">x</span><span class="p">[</span><span class="s2">"Predicted"</span><span class="p">],</span>
            <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span>
        <span class="p">),</span>
        <span class="n">include_groups</span><span class="o">=</span><span class="kc">False</span>
    <span class="p">)</span>
    <span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">name</span><span class="o">=</span><span class="s2">"Recall"</span><span class="p">)</span>
<span class="p">)</span>

<span class="n">scholarship_recall</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">group_results</span>
    <span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s2">"scholarship"</span><span class="p">)</span>
    <span class="o">.</span><span class="n">apply</span><span class="p">(</span>
        <span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">recall_score</span><span class="p">(</span>
            <span class="n">x</span><span class="p">[</span><span class="s2">"Actual"</span><span class="p">],</span>
            <span class="n">x</span><span class="p">[</span><span class="s2">"Predicted"</span><span class="p">],</span>
            <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span>
        <span class="p">),</span>
        <span class="n">include_groups</span><span class="o">=</span><span class="kc">False</span>
    <span class="p">)</span>
    <span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">name</span><span class="o">=</span><span class="s2">"Recall"</span><span class="p">)</span>
<span class="p">)</span>

<span class="n">scholarship_recall</span><span class="p">[</span><span class="s2">"scholarship"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">scholarship_recall</span><span class="p">[</span><span class="s2">"scholarship"</span><span class="p">]</span>
    <span class="o">.</span><span class="n">replace</span><span class="p">({</span><span class="mi">0</span><span class="p">:</span> <span class="s2">"No"</span><span class="p">,</span> <span class="mi">1</span><span class="p">:</span> <span class="s2">"Yes"</span><span class="p">})</span>
<span class="p">)</span>

<span class="n">age_recall</span><span class="p">[</span><span class="s2">"Recall"</span><span class="p">]</span> <span class="o">=</span> <span class="n">age_recall</span><span class="p">[</span><span class="s2">"Recall"</span><span class="p">]</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
<span class="n">scholarship_recall</span><span class="p">[</span><span class="s2">"Recall"</span><span class="p">]</span> <span class="o">=</span> <span class="n">scholarship_recall</span><span class="p">[</span><span class="s2">"Recall"</span><span class="p">]</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>

<span class="n">display</span><span class="p">(</span><span class="n">Markdown</span><span class="p">(</span><span class="s2">"#### Recall by Age Group"</span><span class="p">))</span>
<span class="n">display</span><span class="p">(</span><span class="n">age_recall</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="s2">"Age Group"</span><span class="p">:</span> <span class="s2">"Age Group"</span><span class="p">}))</span>

<span class="n">display</span><span class="p">(</span><span class="n">Markdown</span><span class="p">(</span><span class="s2">"#### Recall by Scholarship Status"</span><span class="p">))</span>
<span class="n">display</span><span class="p">(</span>
    <span class="n">scholarship_recall</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span>
        <span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="s2">"scholarship"</span><span class="p">:</span> <span class="s2">"Scholarship"</span><span class="p">}</span>
    <span class="p">)</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_markdown rendered_html output_subarea">
<h4 id="Recall-by-Age-Group">Recall by Age Group<a class="anchor-link" href="#Recall-by-Age-Group">¶</a></h4>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_html rendered_html output_subarea">
<div class="colab-df-container" id="df-ba86c89d-06c4-4bbf-9f02-20189bc685e0">
<div>
<style scoped="">
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
<th>Age Group</th>
<th>Recall</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>0-17</td>
<td>0.638</td>
</tr>
<tr>
<th>1</th>
<td>18-34</td>
<td>0.665</td>
</tr>
<tr>
<th>2</th>
<td>35-49</td>
<td>0.648</td>
</tr>
<tr>
<th>3</th>
<td>50-64</td>
<td>0.423</td>
</tr>
<tr>
<th>4</th>
<td>65+</td>
<td>0.403</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-ba86c89d-06c4-4bbf-9f02-20189bc685e0')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-ba86c89d-06c4-4bbf-9f02-20189bc685e0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ba86c89d-06c4-4bbf-9f02-20189bc685e0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_markdown rendered_html output_subarea">
<h4 id="Recall-by-Scholarship-Status">Recall by Scholarship Status<a class="anchor-link" href="#Recall-by-Scholarship-Status">¶</a></h4>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_html rendered_html output_subarea">
<div class="colab-df-container" id="df-71cb22c8-7e40-4bd1-91f9-c6dfd9d1eab3">
<div>
<style scoped="">
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
<th>Scholarship</th>
<th>Recall</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>No</td>
<td>0.561</td>
</tr>
<tr>
<th>1</th>
<td>Yes</td>
<td>0.773</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-71cb22c8-7e40-4bd1-91f9-c6dfd9d1eab3')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-71cb22c8-7e40-4bd1-91f9-c6dfd9d1eab3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-71cb22c8-7e40-4bd1-91f9-c6dfd9d1eab3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Recall-Varied-Across-Patient-Groups">Key Finding: Recall Varied Across Patient Groups<a class="anchor-link" href="#Key-Finding:-Recall-Varied-Across-Patient-Groups">¶</a></h3><p>I found meaningful differences in no-show recall across patient groups.</p>
<p>Recall was relatively similar for patients aged <strong>0–17 (0.638)</strong>, <strong>18–34 (0.665)</strong>, and <strong>35–49 (0.648)</strong>, but it dropped substantially for older patients. Recall decreased to <strong>0.423 for ages 50–64</strong> and <strong>0.403 for ages 65+</strong>, indicating that the model was less effective at identifying no-shows among older patients.</p>
<p>I also found a difference by scholarship status. Recall was 0.773 among patients enrolled in the scholarship program compared with 0.561 among those who were not enrolled.</p>
<p>These results suggest that overall model performance does not fully reflect how consistently the model performs across different patient groups. I therefore interpret the model's predictions cautiously and consider subgroup performance an important area for further validation before practical deployment.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred</span><span class="p">)</span>

<span class="n">disp</span> <span class="o">=</span> <span class="n">ConfusionMatrixDisplay</span><span class="p">(</span>
    <span class="n">confusion_matrix</span><span class="o">=</span><span class="n">cm</span><span class="p">,</span>
    <span class="n">display_labels</span><span class="o">=</span><span class="p">[</span><span class="s2">"Show"</span><span class="p">,</span> <span class="s2">"No-show"</span><span class="p">]</span>
<span class="p">)</span>

<span class="n">disp</span><span class="o">.</span><span class="n">plot</span><span class="p">()</span>

<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Confusion Matrix - Random Forest (Threshold = 0.24)"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Predicted Appointment Outcome"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Actual Appointment Outcome"</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img alt="No description has been provided for this image" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAj8AAAHHCAYAAABQhTneAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAfptJREFUeJzt3XdUFNfbB/Dv0nuVqogoiqCosSsqqChRYy+xJELsxq4xlsTesZeoseIvQWOMSuyxgg27WLETO2ChCCpl975/8DJxBZR1l+hmv59z5hx25s7dZ4aBffaWGZkQQoCIiIhIR+h97ACIiIiI/k1MfoiIiEinMPkhIiIincLkh4iIiHQKkx8iIiLSKUx+iIiISKcw+SEiIiKdwuSHiIiIdAqTHyIiItIpTH4+sps3b6Jp06awtraGTCZDRESERuv/+++/IZPJEBYWptF6tVlAQAACAgI+dhifDF4jH0/z5s3Ru3dvlffL/Z3NmTOnCKJSXVHEExYWBplMhr///vu9ZUuVKoWQkBCNvbc2evbsGczNzbFr166PHYpWYPID4Pbt2+jbty9Kly4NExMTWFlZwc/PDwsXLsSrV6+K9L2Dg4Nx6dIlTJs2Db/88guqV69epO/3bwoJCYFMJoOVlVW+5/HmzZuQyWQf/E/z0aNHmDhxImJiYjQQ7b+jVKlS0jHLZDKYm5ujZs2a+N///vexQ/ukvH2e3lxev379scPL4/jx45g4cSKSk5MLvc+xY8ewd+9ejBo1CsC7j/nNhUnqp+P48eOoV68ezMzM4OzsjMGDByMtLe29+92/fx+TJk1CzZo1YWtri2LFiiEgIAD79+9/7769e/eGTCbDF198obTe3t4evXr1wrhx4z74eHSJwccO4GPbuXMnOnbsCGNjY3Tv3h0VK1ZEZmYmjh49ipEjR+LKlStYsWJFkbz3q1evEB0djR9++AEDBw4skvdwd3fHq1evYGhoWCT1v4+BgQFevnyJ7du3o1OnTkrbwsPDYWJi8sEfZo8ePcKkSZNQqlQpVKlSpdD77d2794PeT1OqVKmCESNGAAAeP36MVatWITg4GBkZGR/UCvBf9eZ5epORkdFHiObdjh8/jkmTJiEkJAQ2NjaF2mf27Nlo3LgxPD09AQALFixQ+uDctWsXNmzYgPnz56NYsWLS+rp162o0dvowMTExaNy4Mby9vTFv3jw8ePAAc+bMwc2bN7F79+537vvnn39i1qxZaNOmDYKDg5GdnY3//e9/aNKkCdasWYNvvvkm3/3OnDmDsLAwmJiY5Lu9X79+WLRoEQ4ePIhGjRqpfYz/ZTqd/MTFxaFz585wd3fHwYMH4eLiIm0bMGAAbt26hZ07dxbZ+z958gQACv3P8kPIZLIC/1D+DcbGxvDz88OGDRvyJD/r169HixYtsHnz5n8llpcvX8LMzOyjf3gWL14cX331lfQ6JCQEpUuXxvz585n8vOHt86QpCoUCmZmZH/XvIjExETt37sTy5culdW3atFEqEx8fjw0bNqBNmzYoVaqU0rbCdAW9S3p6OszNzdWqQ9eNHTsWtra2iIyMhJWVFYCc1rvevXtj7969aNq0aYH7NmzYEPfu3VNKavv164cqVapg/Pjx+SY/QggMHjwY3bt3x4EDB/Kt19vbGxUrVkRYWBiTn/fQ6W6v0NBQpKWlYfXq1UqJTy5PT08MGTJEep2dnY0pU6agTJkyMDY2RqlSpTB27FhkZGQo7VeqVCl88cUXOHr0KGrWrAkTExOULl1aqWtj4sSJcHd3BwCMHDkSMplM+gcXEhKS559d7j4ymUxp3b59+1CvXj3Y2NjAwsICXl5eGDt2rLS9oPEcBw8eRP369WFubg4bGxu0bt0asbGx+b7frVu3pG+01tbW+Oabb/Dy5cuCT+xbunbtit27dyt1CZw+fRo3b95E165d85R//vw5vvvuO/j6+sLCwgJWVlZo1qwZLly4IJWJjIxEjRo1AADffPNNni6BgIAAVKxYEWfPnkWDBg1gZmYmnZe3x/wEBwfDxMQkz/EHBQXB1tYWjx49KvSxfggHBweUL18et2/fVlp/5MgRdOzYESVLloSxsTHc3NwwbNiwPF2IISEhsLCwwMOHD9GmTRtYWFjAwcEB3333HeRyuVLZ5ORkhISEwNraGjY2NggODi6wq0aVa+TGjRv46quvYG1tDQcHB4wbNw5CCNy/fx+tW7eGlZUVnJ2dMXfuXPVP2P9LT0/HiBEj4ObmBmNjY3h5eWHOnDkQQiiVk8lkGDhwIMLDw1GhQgUYGxtjz549AICHDx+iR48ecHJygrGxMSpUqIA1a9bkea/FixejQoUKMDMzg62tLapXr47169dL52DkyJEAAA8PD+lafFeCsnPnTmRnZyMwMFDt87BixQrpf1KNGjVw+vRppe2518ft27fRvHlzWFpaolu3bgByEsEFCxagQoUKMDExgZOTE/r27YukpCSlOs6cOYOgoCAUK1YMpqam8PDwQI8ePT4oHqBw11Z+hBCYOnUqSpQoATMzMzRs2BBXrlwp7KnSmNTUVOzbtw9fffWVlPgAQPfu3WFhYYHff//9nftXqFBBKfEBcr4oNm/eHA8ePMCLFy/y7PPLL7/g8uXLmDZt2jvrbtKkCbZv357n74CU6XTLz/bt21G6dOlCNyP36tUL69atQ4cOHTBixAicPHkSM2bMQGxsLLZu3apU9tatW+jQoQN69uyJ4OBgrFmzBiEhIahWrRoqVKiAdu3awcbGBsOGDUOXLl3QvHlzWFhYqBT/lStX8MUXX6BSpUqYPHkyjI2NcevWLRw7duyd++3fvx/NmjVD6dKlMXHiRLx69QqLFy+Gn58fzp07lyfx6tSpEzw8PDBjxgycO3cOq1atgqOjI2bNmlWoONu1a4d+/fphy5Yt0j/M9evXo3z58qhatWqe8nfu3EFERAQ6duwIDw8PJCQk4Oeff4a/vz+uXr0KV1dXeHt7Y/LkyRg/fjz69OmD+vXrA1DuEnj27BmaNWuGzp0746uvvoKTk1O+8S1cuBAHDx5EcHAwoqOjoa+vj59//hl79+7FL7/8AldX10Id54fKzs7GgwcPYGtrq7R+06ZNePnyJfr37w97e3ucOnUKixcvxoMHD7Bp0yalsnK5HEFBQahVqxbmzJmD/fv3Y+7cuShTpgz69+8PIOeDo3Xr1jh69Cj69esHb29vbN26FcHBwXliUvUa+fLLL+Ht7Y2ZM2di586dmDp1Kuzs7PDzzz+jUaNGmDVrFsLDw/Hdd9+hRo0aaNCgwXvPS1ZWFp4+faq0zszMDGZmZhBCoFWrVjh06BB69uyJKlWq4K+//sLIkSPx8OFDzJ8/X2m/gwcP4vfff8fAgQNRrFgxlCpVCgkJCahdu7aUHDk4OGD37t3o2bMnUlNTMXToUADAypUrMXjwYHTo0AFDhgzB69evcfHiRZw8eRJdu3ZFu3btcOPGjTxdVA4ODgUe2/Hjx2Fvby99AfpQ69evx4sXL9C3b1/IZDKEhoaiXbt2uHPnjlJXd3Z2NoKCglCvXj3MmTMHZmZmAIC+ffsiLCwM33zzDQYPHoy4uDgsWbIE58+fx7Fjx2BoaIjExEQ0bdoUDg4OGD16NGxsbPD3339jy5YtHxSPqtfWm8aPH4+pU6eiefPmaN68Oc6dO4emTZsiMzOzUOcrKSkpzxeC/OReZwW5dOkSsrOz84zRNDIyQpUqVXD+/PlCxfO2+Pj4fN/7xYsXGDVqFMaOHQtnZ+d31lGtWjXMnz8fV65cQcWKFT8oDp0gdFRKSooAIFq3bl2o8jExMQKA6NWrl9L67777TgAQBw8elNa5u7sLAOLw4cPSusTERGFsbCxGjBghrYuLixMAxOzZs5XqDA4OFu7u7nlimDBhgnjzVzZ//nwBQDx58qTAuHPfY+3atdK6KlWqCEdHR/Hs2TNp3YULF4Senp7o3r17nvfr0aOHUp1t27YV9vb2Bb7nm8dhbm4uhBCiQ4cOonHjxkIIIeRyuXB2dhaTJk3K9xy8fv1ayOXyPMdhbGwsJk+eLK07ffp0nmPL5e/vLwCI5cuX57vN399fad1ff/0lAIipU6eKO3fuCAsLC9GmTZv3HqOq3N3dRdOmTcWTJ0/EkydPxKVLl8TXX38tAIgBAwYolX358mWe/WfMmCFkMpm4e/eutC44OFgAUDo3Qgjx2WefiWrVqkmvIyIiBAARGhoqrcvOzhb169dX+xrp06ePUp0lSpQQMplMzJw5U1qflJQkTE1NRXBwcKHOE4A8y4QJE5SOZerUqUr7dejQQchkMnHr1i1pHQChp6cnrly5olS2Z8+ewsXFRTx9+lRpfefOnYW1tbV0/lu3bi0qVKjwznhnz54tAIi4uLj3HpsQQtSrV0/pd6Nqnbl/N/b29uL58+fS+j///FMAENu3b5fW5V4fo0ePVqrjyJEjAoAIDw9XWr9nzx6l9Vu3bhUAxOnTpwuMVZV4CnttrV27Vun4ExMThZGRkWjRooVQKBRSubFjxwoAal1XBV1nBdm0aVOe//G5OnbsKJydnd8by9tu3rwpTExMxNdff51n23fffSc8PDzE69evpeNo0aJFvvUcP35cABAbN25UOQZdorPdXqmpqQAAS0vLQpXPnT44fPhwpfW5AzLfHhvk4+MjtUYAOd8Cvby8cOfOnQ+O+W25Y4X+/PNPKBSKQu3z+PFjxMTEICQkBHZ2dtL6SpUqoUmTJvlOk+zXr5/S6/r16+PZs2fSOSyMrl27IjIyEvHx8Th48CDi4+Pz7fICcpp/9fRyLk25XI5nz55JXXrnzp0r9HsaGxsXOHDwbU2bNkXfvn0xefJktGvXDiYmJvj5558L/V6q2Lt3LxwcHODg4ABfX1/88ssv+OabbzB79mylcqamptLP6enpePr0KerWrQshRL7fLPP7Pb15ve3atQsGBgZSSxAA6OvrY9CgQUr7fcg10qtXL6U6q1evDiEEevbsKa23sbFR6W+gVq1a2Ldvn9LSvXt36Vj09fUxePBgpX1GjBgBIUSeAaf+/v7w8fGRXgshsHnzZrRs2RJCCDx9+lRagoKCkJKSIl1rNjY2ePDgQb7dNx/q2bNneVr6PsSXX36pVE/u/5z8zvGbv3cgp2XR2toaTZo0UTr+atWqwcLCAocOHQLwz/+ZHTt2ICsrS614PuTayrV//35kZmZi0KBBSt3/uS10hREeHp7nmspvyb3OCpLb9WxsbJxnm4mJicqzhF++fImOHTvC1NQUM2fOVNp248YNLFy4ELNnz873/d6We/7fbjUlZTrb7ZXbT5tf32p+7t69Cz09PWlmRi5nZ2fY2Njg7t27SutLliyZpw5bW9s8fenq+PLLL7Fq1Sr06tULo0ePRuPGjdGuXTt06NBBSh7yOw4A8PLyyrPN29sbf/31V57BkG8fS+4fV1JSklJ/97vkjjXYuHEjYmJiUKNGDXh6euY7LkKhUGDhwoVYunQp4uLilJqp7e3tC/V+QM6AWVUGN8+ZMwd//vknYmJisH79ejg6Or53nydPnijFZ2Fh8d7uy1q1amHq1KmQy+W4fPkypk6diqSkpDyx3rt3D+PHj8e2bdvyXDcpKSlKr01MTPJ0s7x9vd29excuLi554nv7WtDENWJtbQ0TE5M84xqsra3x7NmzPPXmp1ixYgWOibl79y5cXV3zfHnx9vZWOoZcHh4eSq+fPHmC5ORkrFixosDZnImJiQCAUaNGYf/+/ahZsyY8PT3RtGlTdO3aFX5+foU6joIIDYzJeNff5psMDAxQokQJpXU3b95ESkpKgdd57vH7+/ujffv2mDRpEubPn4+AgAC0adMGXbt2zfNh/L54PuTaypW7b9myZZXWOzg4FDqRVPd3liv3i8nb4z0B4PXr10pfXN5HLpejc+fOuHr1Knbv3p2nm33IkCGoW7cu2rdvX6j6cq+rt8eHkjKdTn5cXV1x+fJllfYr7AWlr6+f7/rC/MMr6D3e7qs2NTXF4cOHcejQIezcuRN79uzBxo0b0ahRI+zdu7fAGFSlzrHkMjY2Rrt27bBu3TrcuXMHEydOLLDs9OnTMW7cOPTo0QNTpkyBnZ0d9PT0MHTo0EK3cAFQ6R8QAJw/f176h3/p0iV06dLlvfvUqFFD6YN2woQJ7zw2QPlDPSgoCOXLl8cXX3yBhQsXSi2LcrkcTZo0wfPnzzFq1CiUL18e5ubmePjwIUJCQvKcB039rj9Ufu+vietGU96+FnLP31dffZXvmCcgpzUCyPlQvn79Onbs2IE9e/Zg8+bNWLp0KcaPH49JkyZ9UDz29vYa+SJU2HP8ZmtqLoVCAUdHR4SHh+dbR24yLZPJ8Mcff+DEiRPYvn07/vrrL/To0QNz587FiRMnlJLpT+l3np+3v6wU5H1fYnInyDx+/DjPtsePH6s0TrB3797YsWMHwsPD88zQOnjwIPbs2YMtW7YofVHMzs7Gq1ev8Pfff8POzk7pS2judfX2Fw9SprPJDwB88cUXWLFiBaKjo1GnTp13lnV3d4dCocDNmzelb5cAkJCQgOTkZLUHLr7J1tY23xk4b3+bBQA9PT00btwYjRs3xrx58zB9+nT88MMPOHToUL7fmnPjvH79ep5t165dQ7FixYpsCmzXrl2xZs0a6OnpoXPnzgWW++OPP9CwYUOsXr1aaX1ycrLSH7Qmv9mkp6fjm2++gY+PD+rWrYvQ0FC0bdtWmlFWkPDwcKUm7tKlS6v83i1atIC/vz+mT5+Ovn37wtzcHJcuXcKNGzewbt06pSb4ffv2qVx/Lnd3dxw4cABpaWlK/9jfvhY+5jVSWO7u7ti/fz9evHih1Ppz7do1afu7ODg4wNLSEnK5vFAzrszNzfHll1/iyy+/RGZmJtq1a4dp06ZhzJgxMDExUflaLF++/L92i4eClClTBvv374efn1+hvijUrl0btWvXxrRp07B+/Xp069YNv/32m1KX5/uoc23l7nvz5k2lv7MnT54UOpF8+8tKQd73JaZixYowMDDAmTNnlG7hkZmZiZiYmDy39SjIyJEjsXbtWixYsCDfL1v37t0DkDNp5G0PHz6Eh4cH5s+fr9T1FxcXBwBKn1OUl86O+QGA77//Hubm5ujVqxcSEhLybL99+zYWLlwIIKfbBsi5Edmb5s2bByDnA0xTypQpg5SUFFy8eFFa9/jx4zwzyp4/f55n39yb/eXXHAvkfGOpUqUK1q1bp5RgXb58GXv37pWOsyg0bNgQU6ZMwZIlS945Y0FfXz/PN8VNmzbh4cOHSuty/0mqclfdgowaNQr37t3DunXrMG/ePJQqVUq68eC7+Pn5ITAwUFo+JPnJff9nz55h5cqVAP75Bv3meRBCSNfjh2jevDmys7OxbNkyaZ1cLsfixYuVyn3Ma6SwmjdvDrlcjiVLliitnz9/PmQyGZo1a/bO/fX19dG+fXts3rw539bf3HtwAcjTTWdkZAQfHx8IIaQxMKpei3Xq1EFSUpJGxwCqqlOnTpDL5ZgyZUqebdnZ2dKxJCUl5fl7fN//mYKoc20FBgbC0NAQixcvVorn7f/J76KpMT/W1tYIDAzEr7/+qjR04pdffkFaWho6duworXv58iWuXbuWZwzO7NmzMWfOHIwdO1bplipvatSoEbZu3ZpncXBwQPXq1bF161a0bNlSaZ+zZ8/C2toaFSpUKPR50UU63fJTpkwZrF+/Xpqm++Ydno8fP45NmzZJz4upXLkygoODsWLFCiQnJ8Pf3x+nTp3CunXr0KZNGzRs2FBjcXXu3BmjRo1C27ZtMXjwYLx8+RLLli1DuXLllAb8Tp48GYcPH0aLFi3g7u6OxMRELF26FCVKlEC9evUKrH/27Nlo1qwZ6tSpg549e0pTTa2trd/bZaMOPT09/Pjjj+8t98UXX2Dy5Mn45ptvULduXVy6dAnh4eF5EosyZcrAxsYGy5cvh6WlJczNzVGrVq084zve5+DBg1i6dCkmTJggTb1fu3YtAgICMG7cOISGhqpU34do1qwZKlasiHnz5mHAgAEoX748ypQpg++++w4PHz6ElZUVNm/erFZXScuWLeHn54fRo0fj77//ho+PD7Zs2ZJn/BDw8a6RwmrZsiUaNmyIH374AX///TcqV66MvXv34s8//8TQoUNRpkyZ99Yxc+ZMHDp0CLVq1ULv3r3h4+OD58+f49y5c9i/f7/05aJp06ZwdnaGn58fnJycEBsbiyVLlqBFixZSq1O1atUAAD/88AM6d+4MQ0NDtGzZssBWjBYtWsDAwAD79+9Hnz59NHRWVOPv74++fftixowZiImJQdOmTWFoaIibN29i06ZNWLhwITp06IB169Zh6dKlaNu2LcqUKYMXL15g5cqVsLKy+qBE+EOvrdx7V82YMQNffPEFmjdvjvPnz2P37t2F7uLR1JgfAJg2bRrq1q0Lf39/9OnTBw8ePMDcuXPRtGlTfP7551K5U6dOoWHDhkqtSVu3bsX333+PsmXLwtvbG7/++qtS3U2aNIGTkxNKliyZ7/jRoUOHwsnJKc+NMYGc1uGWLVtyzM/7/NvTyz5FN27cEL179xalSpUSRkZGwtLSUvj5+YnFixdLUwuFECIrK0tMmjRJeHh4CENDQ+Hm5ibGjBmjVEaIgqchvj3FuqCp7kIIsXfvXlGxYkVhZGQkvLy8xK+//ppnqvuBAwdE69athaurqzAyMhKurq6iS5cu4saNG3ne4+3p4Pv37xd+fn7C1NRUWFlZiZYtW4qrV68qlcl9v7en0r89BbUgb051L0hBU91HjBghXFxchKmpqfDz8xPR0dH5TlH/888/hY+PjzAwMFA6Tn9//wKnJ79ZT2pqqnB3dxdVq1YVWVlZSuWGDRsm9PT0RHR09DuPQRXvmqIaFhamdAxXr14VgYGBwsLCQhQrVkz07t1bXLhwIc/vs6Dz/Pb1IoQQz549E19//bWwsrIS1tbW4uuvvxbnz5/X+DVSUEzv+r286V3nKdeLFy/EsGHDhKurqzA0NBRly5YVs2fPVpoGLYTI9zYCuRISEsSAAQOEm5ubMDQ0FM7OzqJx48ZixYoVUpmff/5ZNGjQQNjb2wtjY2NRpkwZMXLkSJGSkqJU15QpU0Tx4sWFnp5eof4+WrVqJd3+IT+Fmeqe3/8OvDVV+31/hytWrBDVqlUTpqamwtLSUvj6+orvv/9ePHr0SAghxLlz50SXLl1EyZIlhbGxsXB0dBRffPGFOHPmzAfFI0Thrq38/s/I5XIxadIk6X9DQECAuHz5snB3dy/UVHdNO3LkiKhbt64wMTERDg4OYsCAASI1NVWpzKFDh/Kcg9y/m4KWQ4cOvfN9C/r7iI2NFQDE/v37NXF4/2kyIT6RkWhERDrkyJEjCAgIwLVr1/LMYCL6EEOHDsXhw4dx9uxZtvy8B5MfIqKPpFmzZihRooQ01ovoQz179gzu7u74/fffP4lxeZ86Jj9ERESkU3R6thcRERHpHiY/REREpFOY/BAREZFOYfJDREREOkWnb3L4KVEoFHj06BEsLS05RZGISAsJIfDixQu4uroW+HBpdb1+/RqZmZkaqcvIyAgmJiYaqUvbMPn5RDx69Ahubm4fOwwiIlLT/fv3UaJECY3X+/r1a3i4WyA+8f0PZy0MZ2dnxMXF6WQCxOTnE5F7m/y750rByoK9kfTf5D+t58cOgajIyDNf48qGKUoP29WkzMxMxCfKcfdsKVhZqvc5kfpCAfdqfyMzM5PJD308uV1dVhZ6al/URJ8qfSPd+ydLuqeohy5YWMpgYaneeyig28MrmPwQERFpEblQQK7m7YnlQqGZYLQUkx8iIiItooCAAuplP+rur+3Yv0JEREQ6hS0/REREWkQBBdTttFK/Bu3G5IeIiEiLyIWAXM1nkqu7v7ZjtxcRERG90+HDh9GyZUu4urpCJpMhIiJC2paVlYVRo0bB19cX5ubmcHV1Rffu3fHo0SOlOp4/f45u3brBysoKNjY26NmzJ9LS0pTKXLx4EfXr14eJiQnc3NwQGhqaJ5ZNmzahfPnyMDExga+vL3bt2qXy8TD5ISIi0iK5A57VXVSRnp6OypUr46effsqz7eXLlzh37hzGjRuHc+fOYcuWLbh+/TpatWqlVK5bt264cuUK9u3bhx07duDw4cPo06ePtD01NRVNmzaFu7s7zp49i9mzZ2PixIlYsWKFVOb48ePo0qULevbsifPnz6NNmzZo06YNLl++rNLxyITQ8bavT0Rqaiqsra2RdKM07/ND/1nVx/f/2CEQFRl55mtcXPcDUlJSYGVlpfH6cz8n4q65wFLNz4kXLxTwKP/4g2KVyWTYunUr2rRpU2CZ06dPo2bNmrh79y5KliyJ2NhY+Pj44PTp06hevToAYM+ePWjevDkePHgAV1dXLFu2DD/88APi4+NhZGQEABg9ejQiIiJw7do1AMCXX36J9PR07NixQ3qv2rVro0qVKli+fHmhj4GfskRERDoqNTVVacnIyNBIvSkpKZDJZLCxsQEAREdHw8bGRkp8ACAwMBB6eno4efKkVKZBgwZS4gMAQUFBuH79OpKSkqQygYGBSu8VFBSE6OholeJj8kNERKRFNNnt5ebmBmtra2mZMWOG2vG9fv0ao0aNQpcuXaRWpfj4eDg6OiqVMzAwgJ2dHeLj46UyTk5OSmVyX7+vTO72wuJsLyIiIi2iydle9+/fV+r2MjY2VqverKwsdOrUCUIILFu2TK26ihKTHyIiIh1lZWWlsfFJuYnP3bt3cfDgQaV6nZ2dkZiYqFQ+Ozsbz58/h7Ozs1QmISFBqUzu6/eVyd1eWOz2IiIi0iIKDS2alJv43Lx5E/v374e9vb3S9jp16iA5ORlnz56V1h08eBAKhQK1atWSyhw+fBhZWVlSmX379sHLywu2trZSmQMHDijVvW/fPtSpU0eleJn8EBERaRE5hEYWVaSlpSEmJgYxMTEAgLi4OMTExODevXvIyspChw4dcObMGYSHh0MulyM+Ph7x8fHIzMwEAHh7e+Pzzz9H7969cerUKRw7dgwDBw5E586d4erqCgDo2rUrjIyM0LNnT1y5cgUbN27EwoULMXz4cCmOIUOGYM+ePZg7dy6uXbuGiRMn4syZMxg4cKBKx8NuLyIiIi0iF9DAU91VK3/mzBk0bNhQep2bkAQHB2PixInYtm0bAKBKlSpK+x06dAgBAQEAgPDwcAwcOBCNGzeGnp4e2rdvj0WLFkllra2tsXfvXgwYMADVqlVDsWLFMH78eKV7AdWtWxfr16/Hjz/+iLFjx6Js2bKIiIhAxYoVVToe3ufnE8H7/JAu4H1+6L/s37rPz8Wrjhq5z08ln8Qii/VTx5YfIiIiLaKJMTu6/VhTJj9ERERaRQEZ5JCpXYcuY/8KERER6RS2/BAREWkRhchZ1K1DlzH5ISIi0iJyDXR7qbu/tmO3FxEREekUtvwQERFpEbb8qI/JDxERkRZRCBkUQs3ZXmrur+3Y7UVEREQ6hS0/REREWoTdXupj8kNERKRF5NCDXM2OG7mGYtFWTH6IiIi0iNDAmB/BMT9EREREuoMtP0RERFqEY37Ux+SHiIhIi8iFHuRCzTE/Ov54C3Z7ERERkU5hyw8REZEWUUAGhZptFwrodtMPkx8iIiItwjE/6mO3FxEREekUtvwQERFpEc0MeGa3FxEREWmJnDE/aj7YlN1eRERERLqDLT9ERERaRKGBZ3txthcRERFpDY75UR+THyIiIi2igB7v86MmjvkhIiIincKWHyIiIi0iFzLIhZo3OVRzf23H5IeIiEiLyDUw4FnObi8iIiIi3cGWHyIiIi2iEHpQqDnbS8HZXkRERKQt2O2lPnZ7ERERkU5hyw8REZEWUUD92VoKzYSitZj8EBERaRHN3ORQtzt+dPvoiYiISOew5YeIiEiLaObZXrrd9sHkh4iISIsoIIMC6o754R2eiYiISEuw5Ud9un30REREpHPY8kNERKRFNHOTQ91u+2DyQ0REpEUUQgaFuvf50fGnuut26kdEREQ6hy0/REREWkShgW4vXb/JIZMfIiIiLaKZp7rrdvKj20dPREREOoctP0RERFpEDhnkat6kUN39tR2THyIiIi3Cbi/16fbRExERkc5hyw8REZEWkUP9biu5ZkLRWkx+iIiItAi7vdTH5IeIiEiL8MGm6tPtoyciIiKdw5YfIiIiLSIgg0LNMT+CU92JiIhIW7DbS326ffRERESkc9jyQ0REpEUUQgaFUK/bSt39tR2THyIiIi0i18BT3dXdX9vp9tETERGRzmHLDxERkRZht5f62PJDRESkRRTQ08iiisOHD6Nly5ZwdXWFTCZDRESE0nYhBMaPHw8XFxeYmpoiMDAQN2/eVCrz/PlzdOvWDVZWVrCxsUHPnj2RlpamVObixYuoX78+TExM4ObmhtDQ0DyxbNq0CeXLl4eJiQl8fX2xa9culY4FYPJDRERE75Geno7KlSvjp59+ynd7aGgoFi1ahOXLl+PkyZMwNzdHUFAQXr9+LZXp1q0brly5gn379mHHjh04fPgw+vTpI21PTU1F06ZN4e7ujrNnz2L27NmYOHEiVqxYIZU5fvw4unTpgp49e+L8+fNo06YN2rRpg8uXL6t0PDIhhFDxHFARSE1NhbW1NZJulIaVJXNS+m+qPr7/xw6BqMjIM1/j4rofkJKSAisrK43Xn/s50f9IOxhbGKpVV0ZaFpbV3/JBscpkMmzduhVt2rQBkNPq4+rqihEjRuC7774DAKSkpMDJyQlhYWHo3LkzYmNj4ePjg9OnT6N69eoAgD179qB58+Z48OABXF1dsWzZMvzwww+Ij4+HkZERAGD06NGIiIjAtWvXAABffvkl0tPTsWPHDime2rVro0qVKli+fHmhj4GfskRERFokd8yPuoumxMXFIT4+HoGBgdI6a2tr1KpVC9HR0QCA6Oho2NjYSIkPAAQGBkJPTw8nT56UyjRo0EBKfAAgKCgI169fR1JSklTmzffJLZP7PoXFAc9ERERaRGjgqe7i//dPTU1VWm9sbAxjY2OV6oqPjwcAODk5Ka13cnKStsXHx8PR0VFpu4GBAezs7JTKeHh45Kkjd5utrS3i4+Pf+T6FpdbZe7Mvj4iIiLSLm5sbrK2tpWXGjBkfO6R/hcrJj0KhwJQpU1C8eHFYWFjgzp07AIBx48Zh9erVGg+QiIiI/iGHTCMLANy/fx8pKSnSMmbMGJXjcXZ2BgAkJCQorU9ISJC2OTs7IzExUWl7dnY2nj9/rlQmvzrefI+CyuRuLyyVk5+pU6ciLCwMoaGhSv1yFStWxKpVq1StjoiIiFSgEJoY95NTl5WVldKiapcXAHh4eMDZ2RkHDhyQ1qWmpuLkyZOoU6cOAKBOnTpITk7G2bNnpTIHDx6EQqFArVq1pDKHDx9GVlaWVGbfvn3w8vKCra2tVObN98ktk/s+haVy8vO///0PK1asQLdu3aCvry+tr1y5sjQam4iIiP470tLSEBMTg5iYGAA5g5xjYmJw7949yGQyDB06FFOnTsW2bdtw6dIldO/eHa6urtKMMG9vb3z++efo3bs3Tp06hWPHjmHgwIHo3LkzXF1dAQBdu3aFkZERevbsiStXrmDjxo1YuHAhhg8fLsUxZMgQ7NmzB3PnzsW1a9cwceJEnDlzBgMHDlTpeFQe8Pzw4UN4enrmWa9QKJSytU/d21P16NN36YQ5Ni11xM1LZnieYIgJq+NQt1kKACA7Cwib5YLTB63w+K4RzK0U+Kz+C/Qc+wj2ztlSHROCPXD7iimSnxnA0lqeU+YH5TK5HsYZYUBTL+jpA1uuXVLatmWlA3aus0fiIyNY2Waj/hfJ6DHmMYxMeOcI0pw+DU+jT8OzSuv+fmKDDos7w8r0Nfo2PIPanvfhZJ2G5HRTRF4rhWUHaiA9459v7zVKP0C/Rqfh6fQcrzINsDPGC0sP1IRckfe7bwm7FIT3/wMKhQwNZ/Qo8uOjD6PQwIBnVfc/c+YMGjZsKL3OTUiCg4MRFhaG77//Hunp6ejTpw+Sk5NRr1497NmzByYmJtI+4eHhGDhwIBo3bgw9PT20b98eixYtkrZbW1tj7969GDBgAKpVq4ZixYph/PjxSvcCqlu3LtavX48ff/wRY8eORdmyZREREYGKFSuqdDwqJz8+Pj44cuQI3N3dldb/8ccf+Oyzz1Strsg8efIE48ePx86dO5GQkABbW1tUrlwZ48ePh5+f38cOjz7A65d6KF3hFYK6PMfknsozAjJe6eHWJTN0HZqA0j6vkJaij2Xji2NCSGks2XNDKlfZLw2dByfAzikLTx8bYuXk4pjS2wMLtivfiTQ7C5j5bSlUrJWOq2fMlbYd3GKDNdNdMHzuPfjUeImHt40xZ1hJyGRA34mPiu4EkE66nWCLb9e1lF5nK3LGajhYvoSDZToW/FUHdxJt4WKThjEtD8PB8iVGbWwKACjr9BQLv9qFNYerYsKWRnC0SseYloehpyew8C/lbgJ9PTmmddyPmLvOqOSmPKaCPi0KyKCAmo+3UHH/gIAAvOu2gDKZDJMnT8bkyZMLLGNnZ4f169e/830qVaqEI0eOvLNMx44d0bFjx3cH/B4qJz/jx49HcHAwHj58CIVCgS1btuD69ev43//+p3TToY+tffv2yMzMxLp161C6dGkkJCTgwIEDePbs2ccOjT5QjUYvUKPRi3y3mVspMHPjbaV1A6Y9wODmXkh8YAjHEjmtku36PJG2O5XIwpcDEzCphweyswCDN+4ZFjbLBW6er1GlXlqe5OfqGXNUqJGORu2SAQDObpkIaJOE6+fMNHCURMqyFXp4lpb32rqdaIfvNwZJrx8mWWPpgZqY0v4A9PUUkCv00MT3Nm4m2GNVZM69VR48t8aivbUxo9M+rDxUDS8z/xm3+W3j07j7xAan7hRn8kP/eSq3m7Vu3Rrbt2/H/v37YW5ujvHjxyM2Nhbbt29HkyZNiiJGlSUnJ+PIkSOYNWsWGjZsCHd3d9SsWRNjxoxBq1atpHJPnz5F27ZtYWZmhrJly2Lbtm1K9URFRaFmzZowNjaGi4sLRo8ejezsnO6RHTt2wMbGBnK5HAAQExMDmUyG0aNHS/v36tULX3311b9wxJSf9FR9yGQC5tbyfLenJunj4BZb+FRPV0p8Yo5a4MgOGwyY/iDf/Xyqp+PmRTNcO5/zgfT4rhFOH7BCjcap+ZYnUkdJ+xTs/u5/iBgajint98PJOv8vAABgYZyJ9AwjqUvLSF+OzGx9pTIZWQYwMZTD2/WfLwLVPR6icYXbmLWzftEcBGmUXMg0suiyD+o0rF+/Pvbt24fExES8fPkSR48eRdOmTTUd2wezsLCAhYUFIiIikJGRUWC5SZMmoVOnTrh48SKaN2+Obt264fnz5wByxjY1b94cNWrUwIULF7Bs2TKsXr0aU6dOBZBzDl68eIHz588DyEmUihUrhsjISKn+qKgoBAQEFNlxUsEyX8uweporAtokwdxSobRt1VQXtCrji44VfPHkkREmro2TtqU+18ecoSXx3YJ7efbL1ahdMrp/9xgj2niiecnKCKnjg0p109BlcGK+5Yk+1OUHTpi4tSEG/dICM3c0gKvtC6zq+SfMjDLzlLU2e4VeAWex9Yy3tC76lhsquSUgyPcm9GQKOFimoVdAzhiiYpYvc/YzfY2JbQ9h0taGSM8wylMvfXpyx/you+gytY4+LS0NqampSsunwMDAAGFhYVi3bh1sbGzg5+eHsWPH4uLFi0rlQkJC0KVLF3h6emL69OlIS0vDqVOnAABLly6Fm5sblixZgvLly6NNmzaYNGkS5s6dC4VCAWtra1SpUkVKdiIjIzFs2DCcP38eaWlpePjwIW7dugV/f/98Y8zIyPgkz91/QXYWMK1vKUAAg2bmbb3p2D8RS/fewPQNt6CnJzB7SEnkdmUvGOmGhm2T4Fs7vcD6Lxy3wG+LnTBw+gP89Nd1jF8dh1P7rRA+36nAfYg+xPGbJXHgShncSrDHiVtuGPJrc1iaZKJJReUuXnPjTCz8ajfuPLHFz4f+eXzAydtuWLS3Nsa0PILj41diy5DfcOxmSQCQHm/wQ+so7LnkifN3Xf+9AyP6yFROfuLi4tCiRQuYm5vD2toatra2sLW1hY2NjTQP/1PQvn17PHr0CNu2bcPnn3+OyMhIVK1aFWFhYVKZSpUqST+bm5vDyspKuglTbGws6tSpA5nsn6ZBPz8/pKWl4cGDnA9Uf39/REZGQgiBI0eOoF27dvD29sbRo0cRFRUFV1dXlC1bNt/4ZsyYoXRXTTc3tyI4C7onN/FJeGiEGb/dzrf1xtpejhJlMlDNPw1jlt3FqQPWiD2b04UVc8wSfyx3RDO3ymjmVhnzR7ghPVUfzdwq468NdgCAdaHOaNw+Cc26PYeH92v4NUvBN2MeY+NiJyjybywi0oi018a4+8waJez++bJkZpSJRV/vRHqGIUZuCIJcodzNFX68MgKmf4Mv5n2FwJkhiIotBQB4+DznYZY1PB7iq7oXcGLCzzgx4WeMaxMFS9NMnJjwM1p9xtuXfIoU0MCzvdQcMK3tVB7w/NVXX0EIgTVr1sDJyUkpOfjUmJiYoEmTJmjSpAnGjRuHXr16YcKECQgJCQEAGBoqPxVXJpNBocKnV0BAANasWYMLFy7A0NAQ5cuXR0BAACIjI5GUlFRgqw8AjBkzRuneBampqUyA1JSb+DyMM0boH7dgZZf/WJ83if//dWdl5nwPWLD9BhTyf67p439ZY9NPjpi/7SbsnXMGTWe80oNMT3nWg97/v37HZAgitZkaZaGEbSp2vchJ1s2NM7G4+05kZeth+PrPkZld0L90GZ6+yBm4H1TpFuKTLXDtcTEAwDcr20L/jevZv3wcuteLQc9VbZGYap5vbfRxCQ3M9hJMflRz4cIFnD17Fl5eXkURT5Hy8fFBREREocp6e3tj8+bNEEJICd6xY8dgaWmJEiVKAPhn3M/8+fOlRCcgIAAzZ85EUlISRowYUWD9H/LwOF33Kl0Pj+L+OWfx941w+7IpLG2yYeeUhSm9PXDrkikm/+8OFHIZnifmXN6WNnIYGglcO2eG6zFmqFgzHRY22Xj8tzHWhTrDpVQGvKvldHOVLKs8RuzGBTPI9IBS5f95jl3tJqnYssIBnhVfoXzVl3gYZ4R1s11Qq0kK9JW/dBOpZUhQNI5cd8fjZAs4WL5E30anoRAy/HXJE+bGmVjSfQdMDLMx7o8gWBhnwcI4J0FPSjeRxnR87ReD4zfdIIQMDX3iEFLvPEb/3kTa/vdT5RZ7b9dECCHD7US7f/dgqdA08VR2TT7VXRupnPzUqFED9+/f/6STn2fPnqFjx47o0aMHKlWqBEtLS5w5cwahoaFo3bp1oer49ttvsWDBAgwaNAgDBw7E9evXMWHCBAwfPhx6ejn/NGxtbVGpUiWEh4djyZIlAIAGDRqgU6dOyMrKemfLD6nuxgUzfN/hnxts/jyxOACgSafn+GpEPE7stQYAfNukvNJ+oX/cQuW6aTA2VeDYbmv8MtcZr1/qwc4xC9UbvsAPQ+7CyLjwTTZdh8ZDJhMIC3XBs3hDWNtlo3aTFISMVu2pwkTv42SVhmkd9sPa7DWS0k1x4Z4zQla0RfJLU1Qr9RC+bjnd9H8O26C0X8t5XfE4Oadbq27Ze+jR4BwMDeS4GW+PERs+x/H/H/dDpKtUTn5WrVqFfv364eHDh6hYsWKerqM3x9F8LBYWFqhVqxbmz5+P27dvIysrC25ubujduzfGjh1bqDqKFy+OXbt2YeTIkahcuTLs7OzQs2dP/Pjjj0rl/P39ERMTI83qsrOzg4+PDxISEj7pBFEbVa6bhr8exRS4/V3bAMDD+zVCN91+Z5m3Nf3yOZp++Vxpnb4B8NWIBHw1gvdCoaI1dlPBtw85+3dxVB/f77119A9r9d4yb9oRUx47Ysq/vyB9NB/jDs//NTLxrls25uPEiRPo2rUr/v77738qkcmk7qHc+96QalJTU2FtbY2kG6VhZanbFyX9d1Uf3/9jh0BUZOSZr3Fx3Q9ISUmBlZWVxuvP/ZxovbcHDM3Vuy1BVnom/my6pshi/dSp3PLTo0cPfPbZZ9iwYcMnP+CZiIiI6G0qJz93797Ftm3b8n24KRERERWtj/Fsr/8alftXGjVqhAsXLhRFLERERPQeat/jRwOzxbSdyi0/LVu2xLBhw3Dp0iX4+vrmGfD85rOziIiIiD41Kic//frlzC7I77H1HPBMRERUtHifH/WpnPyocgdkIiIi0iwmP+rjnGoiIiLSKR+U/ERFRaFly5bw9PSEp6cnWrVqhSNHjmg6NiIiInoLBzyrT+Xk59dff0VgYCDMzMwwePBgDB48GKampmjcuDHWr19fFDESERHR/xP4Z7r7hy66/gxmlcf8TJs2DaGhoRg2bJi0bvDgwZg3bx6mTJmCrl27ajRAIiIi+gfH/KhP5ZafO3fuoGXLlnnWt2rVCnFxcRoJioiIiKioqJz8uLm54cCBA3nW79+/H25ubhoJioiIiPLHMT/qU7nba8SIERg8eDBiYmJQt25dAMCxY8cQFhaGhQsXajxAIiIi+ge7vdSncvLTv39/ODs7Y+7cufj9998BAN7e3ti4cSNat26t8QCJiIiINEnl5AcA2rZti7Zt22o6FiIiInoPtvyoT+Xk5/Tp01AoFKhVq5bS+pMnT0JfXx/Vq1fXWHBERESkTAgZhJrJi7r7azuVBzwPGDAA9+/fz7P+4cOHGDBggEaCIiIiIioqKrf8XL16FVWrVs2z/rPPPsPVq1c1EhQRERHlL/dGherWoctUbvkxNjZGQkJCnvWPHz+GgcEHDSEiIiKiQuJUd/WpnPw0bdoUY8aMQUpKirQuOTkZY8eORZMmTTQaHBEREZGmqdxUM2fOHDRo0ADu7u747LPPAAAxMTFwcnLCL7/8ovEAiYiI6B8c8Kw+lZOf4sWL4+LFiwgPD8eFCxdgamqKb775Bl26dIGhoWFRxEhERET/j1Pd1ady8nP48GHUrVsXffr0UVqfnZ2Nw4cPo0GDBhoLjoiIiJSx5Ud9Ko/5adiwIZ4/f55nfUpKCho2bKiRoIiIiIiKisotP0IIyGR5M8Znz57B3NxcI0ERERFR/oQGur10veWn0MlPu3btAAAymQwhISEwNjaWtsnlcly8eFF60CkREREVDQFACPXr0GWFTn6sra0B5LT8WFpawtTUVNpmZGSE2rVro3fv3pqPkIiIiEiDCp38rF27FgBQqlQpfPfdd+ziIiIi+ggUkEHGOzyrReUxPxMmTCiKOIiIiKgQONtLfSonPx4eHvkOeM51584dtQIiIiIiKkoqJz9Dhw5Vep2VlYXz589jz549GDlypKbiIiIionwohAwy3uRQLSonP0OGDMl3/U8//YQzZ86oHRAREREVTAgNzPbS8eleKt/ksCDNmjXD5s2bNVUdERERUZFQueWnIH/88Qfs7Ow0VR0RERHlgwOe1ady8vPZZ58pDXgWQiA+Ph5PnjzB0qVLNRocERERKWPyoz6Vk582bdoovdbT04ODgwMCAgJQvnx5TcVFRERE+eCAZ/XxPj9ERESkU1ROfh4+fIjNmzfjxo0bAAAvLy+0a9cOxYsX13hwREREpIyzvdSnUvKzdOlSDB8+HJmZmbCysgIApKamYuTIkZg3bx6+/fbbIgmSiIiIcuQkP+qO+dFQMFqq0FPdd+7cicGDB2PgwIF4+PAhkpOTkZycjIcPH+Lbb7/FkCFDsGvXrqKMlYiIiEhthW75mT17NkaPHo2pU6cqrXdxccG8efNgZmaG0NBQNG/eXONBEhERUQ7O9lJfoVt+zp07h6+//rrA7V9//TXOnTunkaCIiIgof0JDiy4rdPIjl8thaGhY4HZDQ0PI5XKNBEVERERUVAqd/FSoUAF//vlngdsjIiJQoUIFjQRFRERE+cvt9lJ30WWFHvMzYMAA9O/fH8bGxujTpw8MDHJ2zc7Oxs8//4wff/yRd3gmIiIqaprot9Lxfq9CJz/BwcG4dOkSBg4ciDFjxqBMmTIQQuDOnTtIS0vD4MGDERISUoShEhERETTRcsOWn8KbM2cOOnTogA0bNuDmzZsAAH9/f3Tu3Bm1a9cukgCJiIiINEnlOzzXrl2biQ4REdFHwjs8q0/l5IeIiIg+Ht7nR32Fnu1FRERE9F/A5IeIiEibCJlmFhXI5XKMGzcOHh4eMDU1RZkyZTBlyhSIN/rPhBAYP348XFxcYGpqisDAQGl8cK7nz5+jW7dusLKygo2NDXr27Im0tDSlMhcvXkT9+vVhYmICNzc3hIaGfvi5KgCTHyIiIi2SO+ZH3UUVs2bNwrJly7BkyRLExsZi1qxZCA0NxeLFi6UyoaGhWLRoEZYvX46TJ0/C3NwcQUFBeP36tVSmW7duuHLlCvbt24cdO3bg8OHD6NOnj7Q9NTUVTZs2hbu7O86ePYvZs2dj4sSJWLFihdrn7U0qJz+NGjVCcnJynvWpqalo1KiRJmIiIiKiT8jx48fRunVrtGjRAqVKlUKHDh3QtGlTnDp1CkBOq8+CBQvw448/onXr1qhUqRL+97//4dGjR4iIiAAAxMbGYs+ePVi1ahVq1aqFevXqYfHixfjtt9/w6NEjAEB4eDgyMzOxZs0aVKhQAZ07d8bgwYMxb948jR6PyslPZGQkMjMz86x//fo1jhw5opGgiIiIqAAafLhXamqq0pKRkZHvW9atWxcHDhzAjRs3AAAXLlzA0aNH0axZMwBAXFwc4uPjERgYKO1jbW2NWrVqITo6GgAQHR0NGxsbVK9eXSoTGBgIPT09nDx5UirToEEDGBkZSWWCgoJw/fp1JCUlffApe1uhZ3tdvHhR+vnq1auIj4+XXsvlcuzZswfFixfXWGBERESUlyZne7m5uSmtnzBhAiZOnJin/OjRo5Gamory5ctDX18fcrkc06ZNQ7du3QBAygmcnJyU9nNycpK2xcfHw9HRUWm7gYEB7OzslMp4eHjkqSN3m62t7Yccbh6FTn6qVKkCmUwGmUyWb/eWqampUt8fERERfdru378PKysr6bWxsXG+5X7//XeEh4dj/fr1qFChAmJiYjB06FC4uroiODj43wpXYwqd/MTFxUEIgdKlS+PUqVNwcHCQthkZGcHR0RH6+vpFEiQRERG9QUM3KbSyslJKfgoycuRIjB49Gp07dwYA+Pr64u7du5gxYwaCg4Ph7OwMAEhISICLi4u0X0JCAqpUqQIAcHZ2RmJiolK92dnZeP78ubS/s7MzEhISlMrkvs4towmFTn7c3d0BAAqFQmNvTkRERKr5GDc5fPnyJfT0lIcJ6+vrSzmBh4cHnJ2dceDAASnZSU1NxcmTJ9G/f38AQJ06dZCcnIyzZ8+iWrVqAICDBw9CoVCgVq1aUpkffvgBWVlZMDQ0BADs27cPXl5eGuvyAj7wDs83b97EoUOHkJiYmCcZGj9+vEYCIyIionx8hKe6t2zZEtOmTUPJkiVRoUIFnD9/HvPmzUOPHj0AADKZDEOHDsXUqVNRtmxZeHh4YNy4cXB1dUWbNm0AAN7e3vj888/Ru3dvLF++HFlZWRg4cCA6d+4MV1dXAEDXrl0xadIk9OzZE6NGjcLly5excOFCzJ8/X80DVqZy8rNy5Ur0798fxYoVg7OzM2Syf7JHmUzG5IeIiOg/ZvHixRg3bhy+/fZbJCYmwtXVFX379lX6zP/++++Rnp6OPn36IDk5GfXq1cOePXtgYmIilQkPD8fAgQPRuHFj6OnpoX379li0aJG03draGnv37sWAAQNQrVo1FCtWDOPHj1e6F5AmyIRQ7VZH7u7u+PbbbzFq1CiNBqLrUlNTYW1tjaQbpWFlyXtP0n9T9fH9P3YIREVGnvkaF9f9gJSUlEKNo1FV7ueE2/KJ0DM1ef8O76B49Rr3+00sslg/dSq3/CQlJaFjx45FEQsRERG9z0fo9vqvUbmJoWPHjti7d29RxEJERERU5FRu+fH09MS4ceNw4sQJ+Pr6SqOxcw0ePFhjwREREdFb2PKjNpWTnxUrVsDCwgJRUVGIiopS2iaTyZj8EBERFaUPeCp7vnXoMJWTn7i4uKKIg4iIiOhf8cHTijIzM3H9+nVkZ2drMh4iIiJ6ByE0s+gylZOfly9fomfPnjAzM0OFChVw7949AMCgQYMwc+ZMjQdIREREb9DgU911lcrJz5gxY3DhwgVERkYq3bgoMDAQGzdu1GhwRERERJqm8pifiIgIbNy4EbVr11a6u3OFChVw+/ZtjQZHREREb+GAZ7WpnPw8efIEjo6Oedanp6crJUNERESkeTKRs6hbhy5TudurevXq2Llzp/Q6N+FZtWoV6tSpo7nIiIiIKC+O+VGbyi0/06dPR7NmzXD16lVkZ2dj4cKFuHr1Ko4fP57nvj9EREREnxqVW37q1auHmJgYZGdnw9fXF3v37oWjoyOio6NRrVq1ooiRiIiIcuWO+VF30WEqt/wAQJkyZbBy5UpNx0JERETvw8dbqO2Dkh8ASExMRGJiIhQKhdL6SpUqqR0UERERUVFROfk5e/YsgoODERsbC/HWLSJlMhnkcrnGgiMiIqK3sOVHbSonPz169EC5cuWwevVqODk5cXo7ERHRv4nJj9pUTn7u3LmDzZs3w9PTsyjiISIiIipSKs/2aty4MS5cuFAUsRAREdH7cLaX2lRu+Vm1ahWCg4Nx+fJlVKxYEYaGhkrbW7VqpbHgiIiISBnv8Kw+lZOf6OhoHDt2DLt3786zjQOeiYiI6FOncrfXoEGD8NVXX+Hx48dQKBRKCxMfIiKiIsbHW6hN5eTn2bNnGDZsGJycnIoiHiIiIqIipXLy065dOxw6dKgoYiEiIqL3kOGfcT8fvHzsg/jIVB7zU65cOYwZMwZHjx6Fr69vngHPgwcP1lhwRERERJr2QbO9LCwsEBUVlecp7jKZjMmPmtqW84WBzPD9BYm0kJNL3McOgajIZCsy/5030sRUdU51V01cHP95ERERfTS8w7PaVB7zM3nyZLx8+TLP+levXmHy5MkaCYqIiIioqKic/EyaNAlpaWl51r98+RKTJk3SSFBERERUAE51V5vK3V5CiHwfZnrhwgXY2dlpJCgiIiLKH+/wrL5CJz+2traQyWSQyWQoV66cUgIkl8uRlpaGfv36FUmQRERERJpS6ORnwYIFEEKgR48emDRpEqytraVtRkZGKFWqFOrUqVMkQRIREdH/44BntRU6+QkODgYAeHh4oG7dunnu70NERET/AiY/alN5zI+/vz8UCgVu3LiBxMREKBQKpe0NGjTQWHBEREREmqZy8nPixAl07doVd+/ehRDKqSOf6k5ERFS0OOBZfSonP/369UP16tWxc+dOuLi45Dvzi4iIiIoI7/CsNpWTn5s3b+KPP/6Ap6dnUcRDRERE78IxP2pT+SaHtWrVwq1bt4oiFiIiIqIip3LLz6BBgzBixAjEx8fn+1T3SpUqaSw4IiIiUsYxP+pTOflp3749AKBHjx7SOplMJt35mQOeiYiIihC7vdTGp7oTERGRTlE5+XF3dy+KOIiIiKgwNNDtxZafQti2bRuaNWsGQ0NDbNu27Z1lW7VqpZHAiIiIKB/s9lJboZKfNm3aID4+Ho6OjmjTpk2B5Tjmh4iIiD51hUp+3nyExduPsyAiIqJ/EVt+1KbymB8iIiL6eDjVXX0q3+QQAKKiotCyZUt4enrC09MTrVq1wpEjRzQdGxEREZHGqZz8/PrrrwgMDISZmRkGDx6MwYMHw9TUFI0bN8b69euLIkYiIiIijVG522vatGkIDQ3FsGHDpHWDBw/GvHnzMGXKFHTt2lWjARIREdEbOOZHbSq3/Ny5cwctW7bMs75Vq1a8ASIREVERyx3zo+6iy1ROftzc3HDgwIE86/fv3w83NzeNBEVERERUVFTu9hoxYgQGDx6MmJgY1K1bFwBw7NgxhIWFYeHChRoPkIiIiN6i4y036lI5+enfvz+cnZ0xd+5c/P777wAAb29vbNy4Ea1bt9Z4gERERPQGjvlR2wfd56dt27Zo27atpmMhIiIiKnIffJPDM2fOIDY2FgDg4+ODatWqaSwoIiIiyh9vcqg+lZOfBw8eoEuXLjh27BhsbGwAAMnJyahbty5+++03lChRQtMxEhERUS52e6lN5dlevXr1QlZWFmJjY/H8+XM8f/4csbGxUCgU6NWrV1HESERERKQxKrf8REVF4fjx4/Dy8pLWeXl5YfHixahfv75GgyMiIiJl7PZSn8rJj5ubG7KysvKsl8vlcHV11UhQREREVAB2e6lN5W6v2bNnY9CgQThz5oy07syZMxgyZAjmzJmj0eCIiIjo0/Dw4UN89dVXsLe3h6mpKXx9fZVyASEExo8fDxcXF5iamiIwMBA3b95UquP58+fo1q0brKysYGNjg549eyItLU2pzMWLF1G/fn2YmJjAzc0NoaGhGj8WlZOfkJAQxMTEoFatWjA2NoaxsTFq1aqFc+fOoUePHrCzs5MWIiIi0jChoUUFSUlJ8PPzg6GhIXbv3o2rV69i7ty5sLW1lcqEhoZi0aJFWL58OU6ePAlzc3MEBQXh9evXUplu3brhypUr2LdvH3bs2IHDhw+jT58+0vbU1FQ0bdoU7u7uOHv2LGbPno2JEydixYoVqp6ld1K522vBggUaDYCIiIgK72OM+Zk1axbc3Nywdu1aaZ2Hh4f0sxACCxYswI8//ijd8Ph///sfnJycEBERgc6dOyM2NhZ79uzB6dOnUb16dQDA4sWL0bx5c8yZMweurq4IDw9HZmYm1qxZAyMjI1SoUAExMTGYN2+eUpKkLpWTn+DgYI29OREREanoI4z52bZtG4KCgtCxY0dERUWhePHi+Pbbb9G7d28AQFxcHOLj4xEYGCjtY21tjVq1aiE6OhqdO3dGdHQ0bGxspMQHAAIDA6Gnp4eTJ0+ibdu2iI6ORoMGDWBkZCSVCQoKwqxZs5CUlKTU0qSOD7rJoVwux9atW5Vucti6dWsYGHzwPROJiIjoX5aamqr0Onc4y9vu3LmDZcuWYfjw4Rg7dixOnz6NwYMHw8jICMHBwYiPjwcAODk5Ke3n5OQkbYuPj4ejo6PSdgMDA9jZ2SmVebNF6c064+PjP17yc+XKFbRq1Qrx8fHSdPdZs2bBwcEB27dvR8WKFTUSGBEREeVDgy0/bm5uSqsnTJiAiRMn5imuUChQvXp1TJ8+HQDw2Wef4fLly1i+fLlW9gipnPz06tULFSpUwJkzZ6QMLCkpCSEhIejTpw+OHz+u8SCJiIgohybH/Ny/fx9WVlbS+vxafQDAxcUFPj4+Suu8vb2xefNmAICzszMAICEhAS4uLlKZhIQEVKlSRSqTmJioVEd2djaeP38u7e/s7IyEhASlMrmvc8togsqzvWJiYjBjxgylpidbW1tMmzYN58+f11hgREREVLSsrKyUloKSHz8/P1y/fl1p3Y0bN+Du7g4gZ/Czs7MzDhw4IG1PTU3FyZMnUadOHQBAnTp1kJycjLNnz0plDh48CIVCgVq1akllDh8+rHQ/wX379sHLy0tjXV7AByQ/5cqVy5OVAUBiYiI8PT01EhQREREV4CNMdR82bBhOnDiB6dOn49atW1i/fj1WrFiBAQMGAABkMhmGDh2KqVOnYtu2bbh06RK6d+8OV1dXtGnTBkBOS9Hnn3+O3r1749SpUzh27BgGDhyIzp07SzdJ7tq1K4yMjNCzZ09cuXIFGzduxMKFCzF8+HA1TlheKnd7zZgxA4MHD8bEiRNRu3ZtAMCJEycwefJkzJo1S2nw1JtNaURERKS+jzHVvUaNGti6dSvGjBmDyZMnw8PDAwsWLEC3bt2kMt9//z3S09PRp08fJCcno169etizZw9MTEykMuHh4Rg4cCAaN24MPT09tG/fHosWLZK2W1tbY+/evRgwYACqVauGYsWKYfz48Rqd5g4AMiGESqdAT++fxiKZTAYgZ37/269lMhnkcrmm4vzPS01NhbW1NQLQGgYyw48dDlGRMHDRXJ890acmW5GJ/fErkJKSUiRf/nM/J7wHToe+scn7d3gHecZrxC4ZW2SxfupUbvk5dOhQUcRBREREhcFne6lN5eTH39+/wG2XL1/mVHciIqKixORHbSoPeH7bixcvsGLFCtSsWROVK1fWRExEREREReaDk5/Dhw8jODgYLi4umDNnDho1aoQTJ05oMjYiIiJ6i0xDiy5TqdsrPj4eYWFhWL16NVJTU9GpUydkZGQgIiIiz82PiIiIqAiw20tthW75admyJby8vHDx4kUsWLAAjx49wuLFi4syNiIiInpL7lR3dRddVuiWn927d2Pw4MHo378/ypYtW5QxERERERWZQrf8HD16FC9evEC1atVQq1YtLFmyBE+fPi3K2IiIiOhtH+EOz/81hU5+ateujZUrV+Lx48fo27cvfvvtN7i6ukKhUGDfvn148eJFUcZJREREuZj4qEXl2V7m5ubo0aMHjh49ikuXLmHEiBGYOXMmHB0d0apVq6KIkYiIiEhj1LrPj5eXF0JDQ/HgwQNs2LBBUzERERFRATjgWX0q3+E5P/r6+mjTpo305FYiIiIqIpzqrja17/BMREREpE000vJDRERE/w5NdFux24uIiIi0B7u91MZuLyIiItIphWr52bZtW6Er5HR3IiKiosNuL/UVKvkp7CwumUwGuVyuTjxERET0Luz2Uluhkh+FQlHUcRAREVFhMPlRG8f8EBERkU75oNle6enpiIqKwr1795CZmam0bfDgwRoJjIiIiPLimB/1qZz8nD9/Hs2bN8fLly+Rnp4OOzs7PH36FGZmZnB0dGTyQ0REVJTY7aU2lbu9hg0bhpYtWyIpKQmmpqY4ceIE7t69i2rVqmHOnDlFESMRERGRxqic/MTExGDEiBHQ09ODvr4+MjIy4ObmhtDQUIwdO7YoYiQiIqL/JxNCI4suUzn5MTQ0hJ5ezm6Ojo64d+8eAMDa2hr379/XbHRERESkTGho0WEqj/n57LPPcPr0aZQtWxb+/v4YP348nj59il9++QUVK1YsihiJiIiINEbllp/p06fDxcUFADBt2jTY2tqif//+ePLkCVasWKHxAImIiOgfubO91F10mcotP9WrV5d+dnR0xJ49ezQaEBEREb0DZ3upjTc5JCIiIp2icsuPh4cHZDJZgdvv3LmjVkBERERUMN7kUH0qJz9Dhw5Vep2VlYXz589jz549GDlypKbiIiIiovyw20ttKic/Q4YMyXf9Tz/9hDNnzqgdEBERERWMLT/q09iYn2bNmmHz5s2aqo6IiIioSHzQg03z88cff8DOzk5T1REREVF+2O2ltg+6yeGbA56FEIiPj8eTJ0+wdOlSjQZHREREeel6t5W6VE5+WrdurZT86OnpwcHBAQEBAShfvrxGgyMiIiLSNJWTn4kTJxZBGERERFQoQuQs6tahw1Qe8Kyvr4/ExMQ86589ewZ9fX2NBEVERET54+Mt1Kdy8iMKyBYzMjJgZGSkdkBERERERanQ3V6LFi0CAMhkMqxatQoWFhbSNrlcjsOHD3PMDxERUVHjbC+1FTr5mT9/PoCclp/ly5crdXEZGRmhVKlSWL58ueYjJCIiIolMkbOoW4cuK3TyExcXBwBo2LAhtmzZAltb2yILioiIiKioqDzb69ChQ0URx78qJCQEycnJiIiI+NihkBoq1kpDx2+foKzvS9g7Z2Nij1KI3mMtbTcxk6PnD49RJygVVrbZiL9vhD9XF8POX4pJZUL/uIXKddOV6t35P3ssGl1Ceu1QPBODZjxAZb80vE7Xx75Ntlgz3QUKecEP+CXShI7f3EHdhgkoUSodmRn6iL1og7WLyuHhXXOlcuV9k9F9wE14VUyBQg7cuWGFcQOrITMjp4X+yx63UaPeU3h4pSI7Sw9fBjRW2t/SOhMjp15EqbJpsLLORPJzY5yIcsC6n8rhVbrG7oVLmsJuL7WpPOC5ffv2mDVrVp71oaGh6Nixo0p1hYSEQCaTYebMmUrrIyIi3vnkeCIAMDFT4M4VEywZWyLf7X0nPkL1gBcIHVQSvf3LY+tKBwyY9hC1m6Yoldv1qx06V/aRllVTXaRtenoCU/4XB0MjgWGtymL2EDc06fQcwSPji/TYiADAt+pz7NxUEiNCauPHb6vBwECBqT+dgbFJtlSmvG8yJi85i/Mn7DGse20M7V4H238vCYXin/+hBoYCR/c7Ydcfbvm+j1DIcCLKEZOHfYbebetj/sSKqFLrOQaOvVLkx0iq42wv9amc/Bw+fBjNmzfPs75Zs2Y4fPiwygGYmJhg1qxZSEpKUnlf0m1nDllhXagLjr/R2vMmn+ovsW+THS5GWyDhgRF2h9vjzlVTeFV5qVQu45Uekp4YSsvLtH/Gs1X1f4GS5V5j1sCSuHPFFGcOWeF/oc5oGfIUBoY63mlORW78oOrYv7047t2xQNxNK8yb4AtHl9fw9E6VyvQecQ3bfiuJTWGlce+OBR7eNcfRfc7Izvrn33v4z56IWF8Kd29Z5vs+aS8MseuPkrgVa40n8aa4cNoeOze5oUKV5KI+RPoQuff5UXfRYSonP2lpaflOaTc0NERqamo+e7xbYGAgnJ2dMWPGjALLbN68GRUqVICxsTFKlSqFuXPnvrfeP/74A76+vjA1NYW9vT0CAwORnq7cvTFnzhy4uLjA3t4eAwYMQFZWlrQtKSkJ3bt3h62tLczMzNCsWTPcvHkTQM6gbwcHB/zxxx9S+SpVqsDF5Z8Wg6NHj8LY2BgvXyp/0NK/5+oZM9RumgJ75ywAApXrpqF46QycjVL+AGjYLgm/X76Mnw9exzdjHsPY9J+kxqf6S/x9zQTJTw2ldWciLWFupYC71+t/61CIAADmFjn/o9JSc65Ha9sMlPdNQcpzI8xZcxK/7j2EmStOwaeKel8m7Yq9Rt2GCbh8jmM76b9J5eTH19cXGzduzLP+t99+g4+Pj8oB6OvrY/r06Vi8eDEePHiQZ/vZs2fRqVMndO7cGZcuXcLEiRMxbtw4hIWFFVjn48eP0aVLF/To0QOxsbGIjIxEu3btlO5RdOjQIdy+fRuHDh3CunXrEBYWplRnSEgIzpw5g23btiE6OhpCCDRv3hxZWVmQyWRo0KABIiMjAeQkSrGxsXj16hWuXbsGAIiKikKNGjVgZmaWb4wZGRlITU1VWkizlv5YHPdumGD9uavYefcipobfwU9ji+PyyX9u03Boqy1CB5bE9x3K4LfFjmjcPgnfL74nbbd1yELSE+UxD7mJkK1DNoj+LTKZQJ/vruNKjA3u3s5J4J2LvwIAdO1zG3u2lsD4QdVw+5oVpi87DVe39HdVl6/vp13A5mP78MtfUXiVboCFUypo9BhIM9jtpT6VR7KNGzcO7dq1w+3bt9GoUSMAwIEDB7BhwwZs2rTpg4Jo27YtqlSpggkTJmD16tVK2+bNm4fGjRtj3LhxAIBy5crh6tWrmD17NkJCQvKt7/Hjx8jOzka7du3g7u4OICdpe5OtrS2WLFkCfX19lC9fHi1atMCBAwfQu3dv3Lx5E9u2bcOxY8dQt25dAEB4eDjc3NwQERGBjh07IiAgAD///DOAnK7Azz77DM7OzoiMjET58uURGRkJf3//Ao95xowZmDRp0gedLyqc1j2eony1lxgfXAqJD4zgWzsdA6Y/xLMEQ5w/kvPhsTvcXir/9zVTPE80QOimO3Bxz8Dju8YfK3SiPPqPjoV7mRcY2bOWtE5PL+cTbPeWEti/vTgA4M51K1Su+QxNWj/EuiXlVHqPlfPKY/3KMihe8iWCB95A7+HXsXSm6l9qqYhxwLPaVG75admyJSIiInDr1i18++23GDFiBB48eID9+/ejTZs2HxzIrFmzsG7dOsTGxiqtj42NhZ+fn9I6Pz8/3Lx5E3K5HEeOHIGFhYW0hIeHo3LlymjcuDF8fX3RsWNHrFy5Ms+YogoVKijdq8jFxUV6bEdsbCwMDAxQq9Y//2Ts7e3h5eUlxefv74+rV6/iyZMniIqKQkBAAAICAhAZGYmsrCwcP34cAQEBBR7vmDFjkJKSIi3379//oPNG+TMyUSBkdDxWTHTFyX3WiIs1xba1xRC1zQYd+j0pcL9r53Ja6lxLZQAAkp4Y5mnhsSmW9f/bOAuG/h39vr+KmvWeYEzfGniWaCKtf/40J0G/f8dCqfz9OAs4OKveLZv0zBgP/rbAycOOWDK9Alp0vA/bYhnqBU/0CVI5+QGAFi1a4NixY0hPT8fTp09x8OBB+Pv74/Llyx8cSIMGDRAUFIQxY8aotF/16tURExMjLa1atYK+vj727duH3bt3w8fHB4sXL4aXl5d0ryIgZ4zSm2QyGRSKwg9g9fX1hZ2dHaKiopSSn6ioKJw+fRpZWVlSq1F+jI2NYWVlpbSQ5hgYCBgaCbz9K1XIAZlewV95ylTM+cB4nphzfVw9Y4ZS5V/D2v6f8WBVG6QhPVUP926Y5FsHkeYI9Pv+Kuo0TMTYftWR8Ei5Gz3hkSmeJhqjeCnlLq7iJdOR+Fi961P2//0ihhzY/8lht5f61P7q+uLFC2zYsAGrVq3C2bNnIZfLP7iumTNnokqVKvDy8pLWeXt749ixY0rljh07hnLlykFfXx+mpqbw9PTMU5dMJoOfnx/8/Pwwfvx4uLu7Y+vWrRg+fPh74/D29kZ2djZOnjwpJTDPnj3D9evXpXFNMpkM9evXx59//okrV66gXr16MDMzQ0ZGBn7++WdUr14d5ubm73obUpOJmRyuHpnSa2e3TJSu8AovkvXx5KERLhw3R+9xj5H5Wg8JDwxRqU46AjskYcUkVwCAi3sGGrZNxqkDlniRZAAPn1foO/ERLkabIy7WFABwLsoS926Y4PvF97B6qitsHbIQMioe28OKISvzg747EBXat6Nj4f/5Y0wZ/hlevTSArX1OK0x6msH/38NHhi3/K4Vu/W4j7oYl7ly3ROOWj1CiVDqmj6oi1ePg/AqWVllwcH4FPT2B0uVyxhg+um+G168MUN3vCWzsMnHzqhVevTSAe5k09BiSM74o8bHpRzhyeic+1V1tH5z8HD58GKtWrcKWLVvg6uqKdu3a4aefflIrGF9fX3Tr1k16jhgAjBgxAjVq1MCUKVPw5ZdfIjo6GkuWLMHSpUsLrOfkyZM4cOAAmjZtCkdHR5w8eRJPnjyBt7d3oeIoW7YsWrdujd69e+Pnn3+GpaUlRo8ejeLFi6N169ZSuYCAAIwYMQLVq1eXnnXWoEEDhIeHY+TIkR94FqiwylV+hdmbb0uv+016BADYu9EWc4eVxIz+7ugx9jFGLbkLSxs5Eh8aIWyWC3b8L2ecT3aWDJ/Vf4G2vZ7AxEyBJ48McXSXNTYscJLqVChkGN/dA4NmPsD87Tfx+qUe9m+yw7rZzv/uwZJOatExpzt81srTSuvnT6wojfH5c0MpGBkr0Hv4dVhaZyHuhiV+HFAd8Q/+aSX6qt8tBLZ8JL1evCEaADC6Tw1cOmuHzAw9fN72AXqPSIOhoQJPE0xw/JATNq31KOpDJPooVEp+4uPjERYWhtWrVyM1NRWdOnVCRkYGIiIiPmimV34mT56sNJusatWq+P333zF+/HhMmTIFLi4umDx5coGDnQHAysoKhw8fxoIFC5Camgp3d3fMnTsXzZo1K3Qca9euxZAhQ/DFF18gMzMTDRo0wK5du5S6y/z9/SGXy5XG9gQEBODPP/9853gf0oyL0RYIcq1c4PakJ4aYO6xkgdufPDLCyPZ5Ww3flvjQCOO+Lv1BMRKpo0W1oEKV2xRWGpvCCr5G50/0xfyJvgVuv3jGHt/1sC9wO31aNNFtpevdXjIhCtf21bJlSxw+fBgtWrRAt27d8Pnnn0NfXx+Ghoa4cOGCxpIfXZWamgpra2sEoDUMZIbv34FICxm4sMWM/ruyFZnYH78CKSkpRTKOM/dzos7nk2FgqN6Yruys14jeM77IYv3UFbrlZ/fu3Rg8eDD69++PsmXLFmVMREREREWm0CM2jx49ihcvXqBatWqoVasWlixZgqdPnxZlbERERPQWzvZSX6GTn9q1a2PlypV4/Pgx+vbti99++w2urq5QKBTYt28fXrx4UZRxEhEREQAohGYWHabyXF1zc3P06NEDR48exaVLlzBixAjMnDkTjo6OaNWqVVHESERERLmEhhYdptaNSry8vBAaGooHDx5gw4YNmoqJiIiIqMho5P78+vr6aNOmjVqPtyAiIqL3k0EDU901Eon24i1qiYiItEnuHZ7VXT7QzJkzIZPJMHToUGnd69evMWDAANjb28PCwgLt27dHQkKC0n737t1DixYtYGZmBkdHR4wcORLZ2crPToyMjETVqlVhbGwMT09PhIWFfXCc78Lkh4iIiArl9OnT+Pnnn1GpUiWl9cOGDcP27duxadMmREVF4dGjR2jXrp20XS6Xo0WLFsjMzMTx48exbt06hIWFYfz48VKZuLg4tGjRAg0bNkRMTAyGDh2KXr164a+//tL4cTD5ISIi0iIfa6p7WloaunXrhpUrV8LW1lZan5KSgtWrV2PevHlo1KgRqlWrhrVr1+L48eM4ceIEAGDv3r24evUqfv31V1SpUgXNmjXDlClT8NNPPyEzM+cZjcuXL4eHhwfmzp0Lb29vDBw4EB06dMD8+fM1ct7exOSHiIhIm2hwtldqaqrSkpGRUeDbDhgwAC1atEBgYKDS+rNnzyIrK0tpffny5VGyZElER+c8Ry46Ohq+vr5wcvrn2YlBQUFITU3FlStXpDJv1x0UFCTVoUlMfoiIiHSUm5sbrK2tpWXGjBn5lvvtt99w7ty5fLfHx8fDyMgINjY2SuudnJwQHx8vlXkz8cndnrvtXWVSU1Px6tWrDzq+gmhkthcRERH9O2RCQKbGgOXcOgDg/v37Ss/2MjY2zlP2/v37GDJkCPbt2wcTE/WeKfapYMsPERGRNlFoaAFgZWWltOSX/Jw9exaJiYmoWrUqDAwMYGBggKioKCxatAgGBgZwcnJCZmYmkpOTlfZLSEiAs3POw4ydnZ3zzP7Kff2+MlZWVjA1Nf2AE1UwJj9ERERUoMaNG+PSpUuIiYmRlurVq6Nbt27Sz4aGhjhw4IC0z/Xr13Hv3j3UqVMHAFCnTh1cunQJiYmJUpl9+/bBysoKPj4+Upk368gtk1uHJrHbi4iISItosturMCwtLVGxYkWldebm5rC3t5fW9+zZE8OHD4ednR2srKwwaNAg1KlTB7Vr1wYANG3aFD4+Pvj6668RGhqK+Ph4/PjjjxgwYIDU2tSvXz8sWbIE33//PXr06IGDBw/i999/x86dO9U61vww+SEiItImmng2l4af7TV//nzo6emhffv2yMjIQFBQEJYuXSpt19fXx44dO9C/f3/UqVMH5ubmCA4OxuTJk6UyHh4e2LlzJ4YNG4aFCxeiRIkSWLVqFYKCgjQbLACZEGqmj6QRqampsLa2RgBaw0Bm+LHDISoSBi7OHzsEoiKTrcjE/vgVSElJURpErCm5nxMN/MbBwEC9gcfZ2a9x+NiUIov1U8cxP0RERKRT2O1FRESkRT70Ds1v16HLmPwQERFpEzUfTCrVocPY7UVEREQ6hS0/REREWkSmyFnUrUOXMfkhIiLSJuz2Uhu7vYiIiEinsOWHiIhIm3yCNznUNkx+iIiItMi//XiL/yJ2exEREZFOYcsPERGRNuGAZ7Ux+SEiItImAoC6U9V1O/dh8kNERKRNOOZHfRzzQ0RERDqFLT9ERETaREADY340EonWYvJDRESkTTjgWW3s9iIiIiKdwpYfIiIibaIAINNAHTqMyQ8REZEW4Wwv9bHbi4iIiHQKW36IiIi0CQc8q43JDxERkTZh8qM2dnsRERGRTmHLDxERkTZhy4/amPwQERFpE051VxuTHyIiIi3Cqe7q45gfIiIi0ils+SEiItImHPOjNiY/RERE2kQhAJmayYtCt5MfdnsRERGRTmHLDxERkTZht5famPwQERFpFQ0kP9Dt5IfdXkRERKRT2PJDRESkTdjtpTYmP0RERNpEIaB2txVnexERERHpDrb8EBERaROhyFnUrUOHMfkhIiLSJhzzozYmP0RERNqEY37UxjE/REREpFPY8kNERKRN2O2lNiY/RERE2kRAA8mPRiLRWuz2IiIiIp3Clh8iIiJtwm4vtTH5ISIi0iYKBQA179Oj0O37/LDbi4iIiHQKW36IiIi0Cbu91Mbkh4iISJsw+VEbu72IiIhIp7Dlh4iISJvw8RZqY/JDRESkRYRQQKj5VHZ199d2TH6IiIi0iRDqt9xwzA8RERGR7mDLDxERkTYRGhjzo+MtP0x+iIiItIlCAcjUHLOj42N+2O1FREREOoUtP0RERNqE3V5qY/JDRESkRYRCAaFmt5euT3VntxcRERHpFCY/RERE2iT32V7qLiqYMWMGatSoAUtLSzg6OqJNmza4fv26UpnXr19jwIABsLe3h4WFBdq3b4+EhASlMvfu3UOLFi1gZmYGR0dHjBw5EtnZ2UplIiMjUbVqVRgbG8PT0xNhYWEfdJrehckPERGRNlEIzSwqiIqKwoABA3DixAns27cPWVlZaNq0KdLT06Uyw4YNw/bt27Fp0yZERUXh0aNHaNeunbRdLpejRYsWyMzMxPHjx7Fu3TqEhYVh/PjxUpm4uDi0aNECDRs2RExMDIYOHYpevXrhr7/+Uv+8vUEmhI6PevpEpKamwtraGgFoDQOZ4ccOh6hIGLg4f+wQiIpMtiIT++NXICUlBVZWVhqvP/dzopFxJxjIjNSqK1tk4mDG7x8c65MnT+Do6IioqCg0aNAAKSkpcHBwwPr169GhQwcAwLVr1+Dt7Y3o6GjUrl0bu3fvxhdffIFHjx7ByckJALB8+XKMGjUKT548gZGREUaNGoWdO3fi8uXL0nt17twZycnJ2LNnj1rH/Ca2/BAREWkTIXLu06PWktPukZqaqrRkZGQUKoSUlBQAgJ2dHQDg7NmzyMrKQmBgoFSmfPnyKFmyJKKjowEA0dHR8PX1lRIfAAgKCkJqaiquXLkilXmzjtwyuXVoCpMfIiIiLSIUQiMLALi5ucHa2lpaZsyY8d73VygUGDp0KPz8/FCxYkUAQHx8PIyMjGBjY6NU1snJCfHx8VKZNxOf3O25295VJjU1Fa9evVL9ZBWAU92JiIi0iVAA0Mwdnu/fv6/U7WVsbPzeXQcMGIDLly/j6NGj6sXwEbHlh4iISEdZWVkpLe9LfgYOHIgdO3bg0KFDKFGihLTe2dkZmZmZSE5OViqfkJAAZ2dnqczbs79yX7+vjJWVFUxNTT/oGPPD5IeIiEiLaLLbq9DvKQQGDhyIrVu34uDBg/Dw8FDaXq1aNRgaGuLAgQPSuuvXr+PevXuoU6cOAKBOnTq4dOkSEhMTpTL79u2DlZUVfHx8pDJv1pFbJrcOTWG3FxERkTbRYLdXYQ0YMADr16/Hn3/+CUtLS2mMjrW1NUxNTWFtbY2ePXti+PDhsLOzg5WVFQYNGoQ6deqgdu3aAICmTZvCx8cHX3/9NUJDQxEfH48ff/wRAwYMkFqc+vXrhyVLluD7779Hjx49cPDgQfz+++/YuXOnesf7FiY/n4jcOw5kI0vtR7YQfbIUmR87AqIik/3/13dR30FGE58T2chSqfyyZcsAAAEBAUrr165di5CQEADA/Pnzoaenh/bt2yMjIwNBQUFYunSpVFZfXx87duxA//79UadOHZibmyM4OBiTJ0+Wynh4eGDnzp0YNmwYFi5ciBIlSmDVqlUICgr6sAMtAO/z84l48OAB3NzcPnYYRESkpvv37yuNh9GU169fw8PDQ2p1UZezszPi4uJgYmKikfq0CZOfT4RCocCjR49gaWkJmUz2scP5z0tNTYWbm1uemQ5E/xW8xv99Qgi8ePECrq6u0NMrmiG1r1+/RmamZlpQjYyMdDLxAdjt9cnQ09Mrkm8K9G65MxyI/qt4jf+7rK2ti7R+ExMTnU1YNImzvYiIiEinMPkhIiIincLkh3SSsbExJkyYUKi7mRJpI17jRAXjgGciIiLSKWz5ISIiIp3C5IeIiIh0CpMfIiIi0ilMfug/SyaTISIi4mOHQfTRhYSEoE2bNh87DKJPBpMf0lpPnjxB//79UbJkSRgbG8PZ2RlBQUE4duzYxw6NKF8hISGQyWSYOXOm0vqIiAje2Z3oX8Tkh7RW+/btcf78eaxbtw43btzAtm3bEBAQgGfPnn3s0IgKZGJiglmzZiEpKeljh0Kks5j8kFZKTk7GkSNHMGvWLDRs2BDu7u6oWbMmxowZg1atWknlnj59irZt28LMzAxly5bFtm3blOqJiopCzZo1YWxsDBcXF4wePRrZ2dkAgB07dsDGxgZyuRwAEBMTA5lMhtGjR0v79+rVC1999dW/cMT0XxEYGAhnZ2fMmDGjwDKbN29GhQoVYGxsjFKlSmHu3LnvrfePP/6Ar68vTE1NYW9vj8DAQKSnpyuVmTNnDlxcXGBvb48BAwYgK+ufJ3snJSWhe/fusLW1hZmZGZo1a4abN28CyHlmlYODA/744w+pfJUqVeDi4iK9Pnr0KIyNjfHy5ctCnwuij4XJD2klCwsLWFhYICIiAhkZGQWWmzRpEjp16oSLFy+iefPm6NatG54/fw4AePjwIZo3b44aNWrgwoULWLZsGVavXo2pU6cCAOrXr48XL17g/PnzAHISpWLFiiEyMlKqPyoqCgEBAUV2nPTfo6+vj+nTp2Px4sV48OBBnu1nz55Fp06d0LlzZ1y6dAkTJ07EuHHjEBYWVmCdjx8/RpcuXdCjRw/ExsYiMjIS7dq1w5u3cTt06BBu376NQ4cOYd26dQgLC1OqMyQkBGfOnMG2bdsQHR0NIQSaN2+OrKwsyGQyNGjQQLr2k5KSEBsbi1evXuHatWsAcv4WatSoATMzM42cJ6IiJYi01B9//CFsbW2FiYmJqFu3rhgzZoy4cOGCtB2A+PHHH6XXaWlpAoDYvXu3EEKIsWPHCi8vL6FQKKQyP/30k7CwsBByuVwIIUTVqlXF7NmzhRBCtGnTRkybNk0YGRmJFy9eiAcPHggA4saNG//G4dJ/QHBwsGjdurUQQojatWuLHj16CCGE2Lp1q8j9d9y1a1fRpEkTpf1GjhwpfHx8Cqz37NmzAoD4+++/C3xfd3d3kZ2dLa3r2LGj+PLLL4UQQty4cUMAEMeOHZO2P336VJiamorff/9dCCHEokWLRIUKFYQQQkRERIhatWqJ1q1bi2XLlgkhhAgMDBRjx44t9Lkg+pjY8kNaq3379nj06BG2bduGzz//HJGRkahatarSt9lKlSpJP5ubm8PKygqJiYkAgNjYWNSpU0dpoKmfnx/S0tKkb+T+/v6IjIyEEAJHjhxBu3bt4O3tjaNHjyIqKgqurq4oW7bsv3PA9J8ya9YsrFu3DrGxsUrrY2Nj4efnp7TOz88PN2/ehFwux5EjR6SWTwsLC4SHh6Ny5cpo3LgxfH190bFjR6xcuTLPmKIKFSpAX19feu3i4qL0t2BgYIBatWpJ2+3t7eHl5SXF5+/vj6tXr+LJkydSi2dAQAAiIyORlZWF48ePsxWUtAaTH9JqJiYmaNKkCcaNG4fjx48jJCQEEyZMkLYbGhoqlZfJZFAoFIWuPyAgAEePHsWFCxdgaGiI8uXLS//wo6Ki4O/vr7FjId3SoEEDBAUFYcyYMSrtV716dcTExEhLq1atoK+vj3379mH37t3w8fHB4sWL4eXlhbi4OGk/df8WfH19YWdnh6ioKKXkJyoqCqdPn0ZWVhbq1q2r0rEQfSxMfug/xcfHJ88gz4J4e3tLYxtyHTt2DJaWlihRogSAf8b9zJ8/X0p0cpOfyMhIftMltcycORPbt29HdHS0tM7b2zvP7RqOHTuGcuXKQV9fH6ampvD09JQWS0tLADnJjJ+fHyZNmoTz58/DyMgIW7duLVQc3t7eyM7OxsmTJ6V1z549w/Xr1+Hj4yPVX79+ffz555+4cuUK6tWrh0qVKiEjIwM///wzqlevDnNzc3VPCdG/gskPaaVnz56hUaNG+PXXX3Hx4kXExcVh06ZNCA0NRevWrQtVx7fffov79+9j0KBBuHbtGv78809MmDABw4cPh55ezp+Gra0tKlWqhPDwcCnRadCgAc6dO4cbN26w5YfU4uvri27dumHRokXSuhEjRuDAgQOYMmUKbty4gXXr1mHJkiX47rvvCqzn5MmTmD59Os6cOYN79+5hy5YtePLkCby9vQsVR9myZdG6dWv07t1baun86quvULx4caW/p4CAAGzYsAFVqlSBhYUF9PT00KBBA4SHh/NvgbQKkx/SShYWFqhVqxbmz5+PBg0aoGLFihg3bhx69+6NJUuWFKqO4sWLY9euXTh16hQqV66Mfv36oWfPnvjxxx+Vyvn7+0Mul0vJj52dHXx8fODs7AwvLy9NHxrpmMmTJyt1P1WtWhW///47fvvtN1SsWBHjx4/H5MmTERISUmAdVlZWOHz4MJo3b45y5crhxx9/xNy5c9GsWbNCx7F27VpUq1YNX3zxBerUqQMhBHbt2qXUXfb23wKQkxC9vY7oUycTb7b5ExEREf3HseWHiIiIdAqTHyIiItIpTH6IiIhIpzD5ISIiIp3C5IeIiIh0CpMfIiIi0ilMfoiIiEinMPkhykdISAjatGkjvQ4ICMDQoUP/9TgiIyMhk8mQnJz8r7/3h9C2eIlINzH5Ia0REhICmUwGmUwGIyMjeHp6YvLkycjOzi7y996yZQumTJlSqLIfKwGYMWMG9PX1MXv27H/1fd9Ut25dPH78GNbW1oXe5+1E82MLCwuDjY1Nocq+evUKEyZMQLly5WBsbIxixYqhY8eOuHLlisrvW6pUKSxYsEDl/YhIdUx+SKt8/vnnePz4MW7evIkRI0Zg4sSJBX7YZ2Zmaux97ezspAdIfqrWrFmD77//HmvWrPloMRgZGcHZ2RkymeyjxfBvycjIQGBgINasWYOpU6fixo0b2LVrF7Kzs1GrVi2cOHHiY4dIRAURRFoiODhYtG7dWmldkyZNRO3atZW2T506Vbi4uIhSpUoJIYS4d++e6Nixo7C2tha2traiVatWIi4uTqojOztbDBs2TFhbWws7OzsxcuRI0b17d6X38vf3F0OGDJFev379Wnz//feiRIkSwsjISJQpU0asWrVKxMXFCQBKS3BwsBBCCLlcLqZPny5KlSolTExMRKVKlcSmTZuUjmfnzp2ibNmywsTERAQEBIi1a9cKACIpKemd5yYyMlIUL15cZGZmCldXV3Hs2DGl7RMmTBCVK1cWy5cvFyVKlBCmpqaiY8eOIjk5Oc/5nThxoihWrJiwtLQUffv2FRkZGUrHPWjQIOHg4CCMjY2Fn5+fOHXqlLT90KFDSvGuXbtWWFtbiz179ojy5csLc3NzERQUJB49eiTF9fb5OnTokHQeN27cKOrVqydMTExE9erVxfXr18WpU6dEtWrVhLm5ufj8889FYmKi0rGuXLlSlC9fXhgbGwsvLy/x008/Sdty6928ebMICAgQpqamolKlSuL48eNK8b+5TJgwId9zPnPmTCGTyURMTIzSerlcLqpXry58fHyEQqEQQuS9foQQonXr1tK14e/vn+d9cx09elT4+/sLU1NTYWNjI5o2bSqeP3+u0u9jz549okqVKsLExEQ0bNhQJCQkiF27dony5csLS0tL0aVLF5Genq50DO+7Vom0GZMf0hr5JT+tWrUSVatWlbZbWFiIr7/+Wly+fFlcvnxZZGZmCm9vb9GjRw9x8eJFcfXqVdG1a1fh5eUlfajPmjVL2Nrais2bN4urV6+Knj17CktLy3cmP506dRJubm5iy5Yt4vbt22L//v3it99+E9nZ2WLz5s0CgLh+/bp4/PixlGBMnTpVlC9fXuzZs0fcvn1brF27VhgbG4vIyEghRE6SZmxsLIYPHy6uXbsmfv31V+Hk5FSo5Ofrr78W3333nRBCiBEjRogePXoobZ8wYYIwNzcXjRo1EufPnxdRUVHC09NTdO3aVen8WlhYiC+//FJcvnxZ7NixQzg4OIixY8dKZQYPHixcXV3Frl27xJUrV0RwcLCwtbUVz549E0Lkn/wYGhqKwMBAcfr0aXH27Fnh7e0tve+LFy9Ep06dxOeffy4eP34sHj9+LDIyMqQkJfd8Xb16VdSuXVtUq1ZNBAQEiKNHj4pz584JT09P0a9fPym+X3/9Vbi4uIjNmzeLO3fuiM2bNws7OzsRFhYmhBBK9e7YsUNcv35ddOjQQbi7u4usrCyRkZEhFixYIKysrKR4Xrx4ke85r1SpkmjatGm+28LDwwUAcf78eSHE+5OfZ8+eiRIlSojJkydL7yuEEOfPnxfGxsaif//+IiYmRly+fFksXrxYPHnyRKXfR+3atZXOmb+/v2jatKk4d+6cOHz4sLC3txczZ86UYnvftUqk7Zj8kNZ4M/lRKBRi3759wtjYWPrQDw4OFk5OTkotFb/88ovw8vKSvoELIURGRoYwNTUVf/31lxBCCBcXFxEaGiptz8rKEiVKlCgw+bl+/boAIPbt25dvnG8nAELkfEM3MzOTWhhy9ezZU3Tp0kUIIcSYMWOEj4+P0vZRo0a9N/lJSUkRpqamUgvE+fPnhYWFhdKH9oQJE4S+vr548OCBtG737t1CT09P+qANDg4WdnZ2Si0Ay5YtExYWFkIul4u0tDRhaGgowsPDpe25LU255y+/5AeAuHXrlrTPTz/9JJycnKTX+SW1uUnKqlWrpHUbNmwQAMSBAwekdTNmzBBeXl7S6zJlyoj169cr1TVlyhRRp06dAuu9cuWKACBiY2OlmK2trfOc57eZmJjkSWhynTt3Tmq5EuL9yY8QQri7u4v58+crlenSpYvw8/PL9z1U+X3s379fKjNjxgwBQNy+fVta17dvXxEUFCSEKNy1SqTtDIquQ41I83bs2AELCwtkZWVBoVCga9eumDhxorTd19cXRkZG0usLFy7g1q1becbrvH79Grdv30ZKSgoeP36MWrVqSdsMDAxQvXp1CCHyjSEmJgb6+vrw9/cvdNy3bt3Cy5cv0aRJE6X1mZmZ+OyzzwAAsbGxSnEAQJ06dd5b94YNG1CmTBlUrlwZAFClShW4u7tj48aN6Nmzp1SuZMmSKF68uFLdCoUC169fh7OzMwCgcuXKMDMzUyqTlpaG+/fvIyUlBVlZWfDz85O2GxoaombNmoiNjS0wPjMzM5QpU0Z67eLigsTExPceFwBUqlRJ+tnJyQlAzu/4zXW5daWnp+P27dvo2bMnevfuLZXJzs7OMwD7zXpdXFwAAImJiShfvnyh4spV0DWiKTExMejYsWO+227fvl3o38fb59HMzAylS5dWWnfq1CkAhbtWibQdkx/SKg0bNsSyZctgZGQEV1dXGBgoX8Lm5uZKr9PS0lCtWjWEh4fnqcvBweGDYjA1NVV5n7S0NADAzp07lRIQADA2Nv6gOHKtXr0aV65cUToXCoUCa9asUUp+PhZDQ0Ol1zKZrNBJw5v75g6ifnudQqEA8M85XrlyZZ4kUl9f/7315tZTWOXKlSsw6ctdX65cOQCAnp5enmPOysp673t8yLWWn7ePN7/fydvnsSiuVaJPBWd7kVYxNzeHp6cnSpYsmSfxyU/VqlVx8+ZNODo6wtPTU2mxtraGtbU1XFxccPLkSWmf7OxsnD17tsA6fX19oVAoEBUVle/23JYnuVwurfPx8YGxsTHu3buXJw43NzcAgLe3t/TtO9f7ZgxdunQJZ86cQWRkJGJiYqQlMjIS0dHRuHbtmlT23r17ePTokVLdenp68PLyktZduHABr169UipjYWEBNzc3lClTBkZGRjh27Ji0PSsrC6dPn4aPj88743wXIyMjpXP1oZycnODq6oo7d+7kOcceHh4aj6dz587Yv38/Lly4oLReoVBg/vz58PHxkVrjHBwc8PjxY6mMXC7H5cuX3/u+lSpVwoEDB/J9/6L6fRTmWiXSdkx+6D+tW7duKFasGFq3bo0jR44gLi4OkZGRGDx4MB48eAAAGDJkCGbOnImIiAhcu3YN33777Tvv0VOqVCkEBwejR48eiIiIkOr8/fffAQDu7u6QyWTYsWMHnjx5grS0NFhaWuK7777DsGHDsG7dOty+fRvnzp3D4sWLsW7dOgBAv379cPPmTYwcORLXr1/H+vXrERYW9s7jW716NWrWrIkGDRqgYsWK0tKgQQPUqFEDq1evlsqamJggODgYFy5cwJEjRzB48GB06tRJ6vICcro2evbsiatXr2LXrl2YMGECBg4cCD09PZibm6N///4YOXIk9uzZg6tXr6J37954+fKlWi1MpUqVwsWLF3H9+nU8ffq0UC0iBZk0aRJmzJiBRYsW4caNG7h06RLWrl2LefPmqRRPWloaDhw4gKdPn+Lly5f5lhs2bBhq1qyJli1bYtOmTbh37x5Onz6N9u3bIzY2FqtXr5ZalRo1aoSdO3di586duHbtGvr375/nGitVqhQOHz6Mhw8f4unTpwCAMWPG4PTp0/j2229x8eJFXLt2DcuWLcPTp0+L7PdRmGuVSOt93CFHRIWX38DYwmx//Pix6N69uyhWrJgwNjYWpUuXFr179xYpKSlCiJwBzkOGDBFWVlbCxsZGDB8+/L1T3V+9eiWGDRsmXFxchJGRkfD09BRr1qyRtk+ePFk4OzsLmUwmDWpVKBRiwYIFwsvLSxgaGgoHBwcRFBQkoqKipP22b98uPD09hbGxsahfv75Ys2ZNgQOeMzIyhL29vdJg7TfNmjVLODo6iszMTGmq+9KlS4Wrq6swMTERHTp0kKZMv3n+xo8fL+zt7YWFhYXo3bu3eP36tdJxDxo0SDqXhZ3q/qatW7cqTeVOTEwUTZo0ERYWFnmmuufOlsqv7oLqDw8PF1WqVBFGRkbC1tZWNGjQQGzZskUIIfKtNykpSXrfXP369RP29vbvnOouhBDp6enihx9+EJ6ensLQ0FDY2dmJ9u3bi0uXLimVy8zMFP379xd2dnbC0dFRzJgxI8+A5+joaFGpUiVhbGysdH4iIyNF3bp1hbGxsbCxsRFBQUHSOVD191HQOcu9PnIV5lol0mYyIYp4xB4RfXQTJ05EREQEYmJiCiwTEhKC5ORkRERE/GtxERF9DOz2IiIiIp3C5IeIiIh0Cru9iIiISKew5YeIiIh0CpMfIiIi0ilMfoiIiEinMPkhIiIincLkh4iIiHQKkx8iIiLSKUx+iIiISKcw+SEiIiKdwuSHiIiIdMr/AeGWZ6wjBMChAAAAAElFTkSuQmCC
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="ROC-Curve">ROC Curve<a class="anchor-link" href="#ROC-Curve">¶</a></h3><p>I also used the ROC curve and ROC-AUC to evaluate how well the final model distinguished between show and no-show appointments.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">RocCurveDisplay</span><span class="o">.</span><span class="n">from_predictions</span><span class="p">(</span>
    <span class="n">y_test</span><span class="p">,</span>
    <span class="n">y_test_prob</span>
<span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"ROC Curve - Random Forest"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"False Positive Rate"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"True Positive Rate"</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img alt="No description has been provided for this image" src="data:image/png;base64,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ffuK999/X/P866+/Fk5OTkKlUgkhhOjSpYtYvny51j52794tHB0d86xBiOxpffT09ISpqakwMjLSjPK/Zs2afLcRQohJkyaJd999N99ac967Xr16WscgIyNDGBsbixMnThS4fyIhhOA9QKrQOnXqhK+++krz3NTUFED22dCKFStw7949JCYmIisrC+np6UhNTYWJiUmu/cycORNjxozB7t27NZfxcmZ2v3HjBm7evIk9e/Zo1hdCQK1WIzQ0FA0aNMiztoSEBJiZmUGtViM9PR2vv/46tm7disTERDx58gTt27fXWr99+/a4ceMGgOzLl2+99Rbq1auHbt26oVevXujatesrHauhQ4di7Nix2LRpEwwNDbFnzx4MGjRIMzHrjRs3cO7cOa0zPpVKVeBxA4B69erhyJEjSE9Px7fffouAgABMmTJFa52NGzdi+/btCA8PR1paGpRKJZo1a1ZgvTdu3EBwcDDMzc21lqenp+Phw4dFOAJU2TAAqUIzNTVFnTp1tJaFhYWhV69emDhxIj755BPY2NjA398fH3zwAZRKZZ5f5IsXL8aQIUNw7Ngx/PLLL/Dx8cG+ffvw9ttvIzk5GePHj8fUqVNzbVe9evV8azM3N8e1a9egp6cHR0dHGBsbAwASExNf+rlatGiB0NBQ/PLLLzh58iQGDBgAT09PHDp06KXb5qd3794QQuDYsWNo3bo1zp49iy+++ELzenJyMpYsWYJ33nkn17ZGRkb57lehUGj+H6xcuRI9e/bEkiVLsGzZMgDAvn37MGvWLHz++edo27YtzM3NsWrVKvz1118F1pucnIyWLVtq/eGRo6w0dKKyjQFIlc7Vq1ehVqvx+eefa85ucu43FcTNzQ1ubm6YMWMGBg8ejB07duDtt99GixYtEBgYmCtoX0ZPTy/PbSwsLODk5IRz586hY8eOmuXnzp1DmzZttNYbOHAgBg4ciP79+6Nbt26Ii4uDjY2N1v5y7repVKoC6zEyMsI777yDPXv2IDg4GPXq1UOLFi00r7do0QJBQUE6f84XLViwAJ07d8bEiRM1n7Ndu3b48MMPNeu8eAanUChy1d+iRQvs378fdnZ2sLCweKWaqHJiIxiqdOrUqYPMzEysX78eISEh2L17NzZv3pzv+mlpaZg8eTLOnDmDv//+G+fOncPly5c1lzbnzJmD8+fPY/LkyQgICMCDBw/w008/6dwI5nkfffQRPv30U+zfvx9BQUHw9vZGQEAApk2bBgBYs2YN9u7di3v37uH+/fs4ePAgHBwc8uy8b2dnB2NjY/j5+SE6OhoJCQn5vu/QoUNx7NgxbN++XdP4JceiRYuwa9cuLFmyBHfu3MHdu3exb98+LFiwQKfP1rZtWzRp0gTLly8HANStWxdXrlzBiRMncP/+fSxcuBCXL1/W2sbV1RU3b95EUFAQYmNjkZmZiaFDh8LW1hZ9+/bF2bNnERoaijNnzmDq1Kl49OiRTjVRJSX1TUiikpJXw4kca9asEY6OjsLY2Fh4eXmJXbt2CQDi2bNnQgjtRioZGRli0KBBwsXFRSgUCuHk5CQmT56s1cDl0qVL4q233hJmZmbC1NRUNGnSJFcjlue92AjmRSqVSixevFg4OzsLAwMD0bRpU/HLL79oXv/mm29Es2bNhKmpqbCwsBBdunQR165d07yO5xrBCCHEli1bhIuLi9DT0xMdO3bM9/ioVCrh6OgoAIiHDx/mqsvPz0+0a9dOGBsbCwsLC9GmTRvxzTff5Ps5fHx8RNOmTXMt37t3rzA0NBTh4eEiPT1djBo1SlhaWgorKysxceJE4e3trbVdTEyM5vgCEKdPnxZCCBEZGSlGjBghbG1thaGhoahVq5YYO3asSEhIyLcmohwyIYSQNoKJiIhKHy+BEhFRpcQAJCKiSokBSERElRIDkIiIKiUGIBERVUoMQCIiqpQYgEREVCkxAImIqFJiABIRUaXEACQiokqJAUhERJUSA5CIiCql/wMSSIGZWBWNQAAAAABJRU5ErkJggg==
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Final-Test-Results">Final Test Results<a class="anchor-link" href="#Final-Test-Results">¶</a></h3><p>I evaluated the final Random Forest model on the untouched holdout test set using the selected classification threshold of 0.24.</p>
<p>I found that the model correctly identified 2,613 of 4,463 actual no-show appointments, corresponding to a no-show recall of approximately 58.5%. The model achieved a precision of approximately 33.0%, meaning that about one-third of the appointments it flagged as high-risk no-shows actually resulted in no-shows.</p>
<p>I also observed 5,294 false-positive predictions, highlighting the trade-off between identifying more potential no-shows and flagging patients who would ultimately attend their appointments.</p>
<p>The model achieved an ROC-AUC of approximately 0.72, indicating moderate ability to distinguish between show and no-show appointments across different classification thresholds. Based on these results, I interpret the model's predictions as risk indicators rather than definitive no-show outcomes.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Operational-Use:-Targeting-the-Highest-Risk-20%25">Operational Use: Targeting the Highest-Risk 20%<a class="anchor-link" href="#Operational-Use:-Targeting-the-Highest-Risk-20%25">¶</a></h3><p>To explore how the model could be used in practice, I simulated a simple intervention strategy in which the hospital contacts only the 20% of appointments with the highest predicted no-show risk.</p>
<p>I evaluated:</p>
<ol>
<li>The no-show rate within this high-risk group, and</li>
<li>The proportion of all actual no-shows captured within this group.</li>
</ol>
<p>This provides a more practical view of how the prediction model could support targeted reminder calls or other interventions when hospital resources are limited.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">risk_analysis</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"y_true"</span><span class="p">:</span> <span class="n">y_test</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">),</span>
    <span class="s2">"no_show_probability"</span><span class="p">:</span> <span class="n">y_test_prob</span>
<span class="p">})</span>

<span class="n">risk_analysis</span> <span class="o">=</span> <span class="n">risk_analysis</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span>
    <span class="s2">"no_show_probability"</span><span class="p">,</span>
    <span class="n">ascending</span><span class="o">=</span><span class="kc">False</span>
<span class="p">)</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">top_20_n</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">risk_analysis</span><span class="p">)</span> <span class="o">*</span> <span class="mf">0.20</span><span class="p">)</span>

<span class="n">top_20</span> <span class="o">=</span> <span class="n">risk_analysis</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">top_20_n</span><span class="p">)</span>

<span class="n">top_20_no_show_rate</span> <span class="o">=</span> <span class="n">top_20</span><span class="p">[</span><span class="s2">"y_true"</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>

<span class="n">total_no_shows</span> <span class="o">=</span> <span class="n">risk_analysis</span><span class="p">[</span><span class="s2">"y_true"</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">captured_no_shows</span> <span class="o">=</span> <span class="n">top_20</span><span class="p">[</span><span class="s2">"y_true"</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>

<span class="n">no_show_capture_rate</span> <span class="o">=</span> <span class="n">captured_no_shows</span> <span class="o">/</span> <span class="n">total_no_shows</span>

<span class="n">operational_summary</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Metric"</span><span class="p">:</span> <span class="p">[</span>
        <span class="s2">"Appointments Targeted"</span><span class="p">,</span>
        <span class="s2">"No-show Rate in Top 20%"</span><span class="p">,</span>
        <span class="s2">"Share of All No-shows Captured"</span>
    <span class="p">],</span>
    <span class="s2">"Result"</span><span class="p">:</span> <span class="p">[</span>
        <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">top_20_n</span><span class="si">:</span><span class="s2">,</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
        <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">top_20_no_show_rate</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="mi">100</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">%"</span><span class="p">,</span>
        <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">no_show_capture_rate</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="mi">100</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">%"</span>
    <span class="p">]</span>
<span class="p">})</span>

<span class="n">operational_summary</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-2d87427f-4988-423e-91f4-560627b7972f">
<div>
<style scoped="">
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
<th>Metric</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Appointments Targeted</td>
<td>4,421</td>
</tr>
<tr>
<th>1</th>
<td>No-show Rate in Top 20%</td>
<td>36.58%</td>
</tr>
<tr>
<th>2</th>
<td>Share of All No-shows Captured</td>
<td>36.23%</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-2d87427f-4988-423e-91f4-560627b7972f')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-2d87427f-4988-423e-91f4-560627b7972f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2d87427f-4988-423e-91f4-560627b7972f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
<div id="id_47d485d6-dade-4bad-9926-0b40c2d3bd73">
<style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
<button class="colab-df-generate" onclick="generateWithVariable('operational_summary')" style="display:none;" title="Generate code using this dataframe.">
<svg height="24px" viewbox="0 0 24 24" width="24px" xmlns="http://www.w3.org/2000/svg">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"></path>
</svg>
</button>
<script>
      (() => {
      const buttonEl =
        document.querySelector('#id_47d485d6-dade-4bad-9926-0b40c2d3bd73 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('operational_summary');
      }
      })();
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Targeting-the-Highest-Risk-20%25">Key Finding: Targeting the Highest-Risk 20%<a class="anchor-link" href="#Key-Finding:-Targeting-the-Highest-Risk-20%25">¶</a></h3><p>I found that the 20% of appointments with the highest predicted no-show risk had a no-show rate of 36.58%, compared with 20.19% across all appointments. This means that the targeted group had a no-show rate approximately 1.8 times higher than the overall rate.</p>
<p>By focusing reminder calls or other interventions on only this highest-risk 20% of appointments, the hospital could reach approximately 36% of all actual no-shows.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="9.-Model-Interpretation">9. Model Interpretation<a class="anchor-link" href="#9.-Model-Interpretation">¶</a></h2><p>To better understand the final Random Forest model, I examined which original predictors contributed most to its ability to distinguish between show and no-show appointments.</p>
<p>I used permutation importance with ROC-AUC as the evaluation metric. This approach measures how much model performance decreases when each predictor is randomly shuffled while keeping the other variables unchanged. Because the permutation is applied to the original input variables, the results remain interpretable at the original feature level rather than being separated into individual one-hot encoded categories.</p>
<p>I also examined <code>neighbourhood</code> separately because it contains many categories and wanted to determine whether including it materially improved predictive performance.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Permutation-Feature-Importance">Permutation Feature Importance<a class="anchor-link" href="#Permutation-Feature-Importance">¶</a></h3><p>I calculated permutation importance on the holdout test set to identify which original variables contributed most to the final Random Forest model's predictive performance.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">permutation_result</span> <span class="o">=</span> <span class="n">permutation_importance</span><span class="p">(</span>
    <span class="n">final_model</span><span class="p">,</span>
    <span class="n">X_test</span><span class="p">,</span>
    <span class="n">y_test</span><span class="p">,</span>
    <span class="n">scoring</span><span class="o">=</span><span class="s2">"roc_auc"</span><span class="p">,</span>
    <span class="n">n_repeats</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
    <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
    <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span>
<span class="p">)</span>

<span class="n">permutation_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Feature"</span><span class="p">:</span> <span class="n">X_test</span><span class="o">.</span><span class="n">columns</span><span class="p">,</span>
    <span class="s2">"Importance"</span><span class="p">:</span> <span class="n">permutation_result</span><span class="o">.</span><span class="n">importances_mean</span>
<span class="p">})</span>

<span class="n">permutation_df</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">permutation_df</span>
    <span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="s2">"Importance"</span><span class="p">,</span> <span class="n">ascending</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="p">)</span>

<span class="n">permutation_df</span><span class="p">[</span><span class="s2">"Feature"</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
    <span class="n">permutation_df</span><span class="p">[</span><span class="s2">"Feature"</span><span class="p">]</span>
    <span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">"_"</span><span class="p">,</span> <span class="s2">" "</span><span class="p">,</span> <span class="n">regex</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">title</span><span class="p">()</span>
<span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">6</span><span class="p">))</span>

<span class="n">plt</span><span class="o">.</span><span class="n">barh</span><span class="p">(</span>
    <span class="n">permutation_df</span><span class="p">[</span><span class="s2">"Feature"</span><span class="p">],</span>
    <span class="n">permutation_df</span><span class="p">[</span><span class="s2">"Importance"</span><span class="p">]</span>
<span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Permutation Feature Importance - Random Forest"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Decrease in ROC-AUC After Permutation"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Feature"</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea">
<img alt="No description has been provided for this image" src="data:image/png;base64,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zZplWNeYMWPUpEkT1axZU506dTIvN+vl5ZXhfTceluDgYHXt2lUjR45UTEyMnnnmGTk6Ouro0aNasGCBPv30U7300ksPZD+UKVNGNWrU0IABA3T+/HnlypVL8+bN040bN+653tdee03z589Xt27dFBUVpdq1ayslJUWHDh3S/PnztXr16kyn5z1MLi4uWrVqlTp06KDq1atr5cqVWr58ud5//31zOlmzZs0UEhKigQMHKj4+XuXLl9eaNWu0ZMkS9e7d2wxed+Lq6qrSpUvr22+/VfHixZUrVy6VLVtWZcuWVd26dTV69Ghdv35dBQsW1Jo1azI8onSvAgICNHDgQA0bNkx16tRRixYt5OzsrJ07d8rHx0cjR46Up6enJk2apNdee02VKlVSmzZtlDdvXp08eVLLly9X7dq10wWph+Htt9/W+PHjFRERoXnz5umZZ56Rk5OTmjVrpq5duyopKUlfffWV8uXLl+5o5b3q16+fZs+ercaNG+vtt982LzdbpEgR7du3z+yXN29eDRgwQOHh4WrcuLGef/55HT58WF988YWqVq36WNys8r333tM333yjJk2aqFevXsqVK5dmzpypEydO6Pvvv7/rze/s7Ow0depUNWnSRGXKlFHHjh1VsGBB/f7774qKipKnp6eWLVumS5cuqVChQnrppZdUvnx5eXh4aN26ddq5c6d5AQkpa7878Yg82otQAfgvS7ts6M6dO+/a9/vvvzeefvppw93d3XB3dzdKlixpvPnmm8bhw4fNPsHBwUaZMmXSrZt22cQxY8ZYtaddAvH2y0tmVFdKSorRv39/I0+ePIabm5sRGhpqHDt2LN3lZg3DML766ivD39/fsLe3t7o045YtW4waNWoYrq6uho+Pj9GvXz/z0o+3XnY0KSnJeOWVV4ycOXMakszLJ2Z0uVnDMIx169YZtWvXNlxdXQ1PT0+jWbNmxsGDB636pF0aNO2Spbe/1tsv3Xq7jC6vmpEpU6YYlStXNlxdXY0cOXIY5cqVM/r162f88ccfZh9b94NhGEZcXJzRsGFDw9nZ2cifP7/x/vvvG2vXrs3wcrMZfSYMwzCuXbtmjBo1yihTpozh7OxsPPXUU0blypWN8PBwIyEhIcv7o0iRIkbTpk3T9ZWU7jKZGX0m08aMi4sznnnmGcPNzc3Inz+/MXjw4HSXab106ZLxv//9z/Dx8TEcHR2NwMBAY8yYMVaX7Mxs22l++ukno3LlyoaTk5PVpWd/++0348UXXzRy5sxpeHl5Ga1atTL++OOPdJenzepnavr06UbFihXNfR0cHGysXbvWqk9UVJQRGhpqeHl5GS4uLkaxYsWMsLAwY9euXRm+hvuR2e+DNGFhYYa9vb152d6lS5caQUFBhouLi+Hn52eMGjXKmD59errXmNn7HxwcbF6WOc2+ffuM4OBgw8XFxShYsKAxbNgwY9q0aRnut4kTJxolS5Y0HB0djfz58xvdu3c3Lly4kG4bGX3Os/KZvN3d9lOauLg446WXXjJy5sxpuLi4GNWqVTN++OEHqz6Z/a5Ns2fPHqNFixZG7ty5DWdnZ6NIkSJG69atjfXr1xuGYRjJyclG3759jfLlyxs5cuQw3N3djfLlyxtffPGF1Th3+p2B7GExjGw4mwcAgP+4sLAwfffddxlOCQGAJxHnWAAAAACwGcECAAAAgM0IFgAAAABsxjkWAAAAAGzGEQsAAAAANiNYAAAAALAZN8gDbpOamqo//vhDOXLkkMViye5yAAAAso1hGLp06ZJ8fHzuehNEggVwmz/++EO+vr7ZXQYAAMBj49SpUypUqNAd+xAsgNvkyJFD0s1/QJ6entlcDQAAQPZJTEyUr6+v+f3oTggWwG3Spj95enoSLAAAAKR7mh7OydsAAAAAbEawAAAAAGAzggUAAAAAmxEsAAAAANiMYAEAAADAZgQLAAAAADYjWAAAAACwGcECAAAAgM0IFgAAAABsRrAAAAAAYDOCBQAAAACbESwAAAAA2IxgAQAAAMBmBAsAAAAANiNYAAAAALAZwQIAAACAzQgWAAAAAGxGsAAAAABgM4IFAAAAAJsRLAAAAADYzCG7CwAeV2UHr5ads1t2lwEAAGAlPqJpdpeQIY5YAAAAALAZwQIAAACAzQgWAAAAAGxGsAAAAABgM4IFAAAAAJsRLAAAAADYjGABAAAAwGYECwAAAAA2I1gAAAAAsBnB4l8qOjpaFotFFy9evGM/Pz8/jR8//pHUBAAAgH8vgkU2+/LLL5UjRw7duHHDbEtKSpKjo6Pq1atn1TctLMTFxd113Fq1aun06dPy8vKSJEVGRipnzpzp+u3cuVNvvPGGTa/hXvj5+clischiscjV1VV+fn5q3bq1NmzY8NC3DQAAgIePYJHNQkJClJSUpF27dpltmzZtkre3t7Zv366rV6+a7VFRUSpcuLCKFSt213GdnJzk7e0ti8Vyx3558+aVm5vb/b+ALBg6dKhOnz6tw4cPa9asWcqZM6caNmyoESNGPJLtAwAA4OEhWGSzEiVKqECBAoqOjjbboqOj9cILL6ho0aLatm2bVXtISIgkafbs2apSpYpy5Mghb29vvfLKKzpz5oxV37SpUNHR0erYsaMSEhLMowZDhgyRlH4qlMVi0dSpU/Xiiy/Kzc1NgYGBWrp0qVXNS5cuVWBgoFxcXBQSEqKZM2fe07SrtFoLFy6sunXrasqUKfrwww81aNAgHT58WJKUkpKiTp06qWjRonJ1dVWJEiX06aefmmP8+OOPcnR01J9//mk1du/evVWnTh1J0q+//qpmzZrpqaeekru7u8qUKaMVK1bc+Y0AAACATQgWj4GQkBBFRUWZz6OiolSvXj0FBweb7f/884+2b99uBovr169r2LBh2rt3rxYvXqz4+HiFhYVlOH6tWrU0fvx4eXp66vTp0zp9+rT69OmTaT3h4eFq3bq19u3bp2effVbt2rXT+fPnJUknTpzQSy+9pObNm2vv3r3q2rWrBg4ceN+v/e2335ZhGFqyZIkkKTU1VYUKFdKCBQt08OBBDRo0SO+//77mz58vSapbt678/f01e/Zsc4zr169r7ty5ev311yVJb775ppKTk/Xjjz9q//79GjVqlDw8PO67RgAAANydQ3YXgJvBonfv3rpx44b++ecf7dmzR8HBwbp+/bq+/PJLSdLWrVuVnJxsBou0L9GS5O/vrwkTJqhq1apKSkpK9yXayclJXl5eslgs8vb2vms9YWFhatu2rSTpo48+0oQJE7Rjxw41btxYkydPVokSJTRmzBhJN4+4HDhw4L6nM+XKlUv58uVTfHy8JMnR0VHh4eHm8qJFi2rr1q2aP3++WrduLUnq1KmTZsyYob59+0qSli1bpqtXr5rLT548qZYtW6pcuXLm/rmT5ORkJScnm88TExPv67UAAAD8l3HE4jFQr149Xb58WTt37tSmTZtUvHhx5c2bV8HBweZ5FtHR0fL391fhwoUlSbt371azZs1UuHBh5ciRQ8HBwZJufqm2VVBQkPmzu7u7PD09zWlWhw8fVtWqVa36V6tWzabtGYZhdS7I559/rsqVKytv3rzy8PDQlClTrF5XWFiYjh07Zk4Ti4yMVOvWreXu7i5J6tWrl4YPH67atWtr8ODB2rdv3x23P3LkSHl5eZkPX19fm14PAADAfxHB4jEQEBCgQoUKKSoqSlFRUWZI8PHxka+vr3766SdFRUWpfv36kqTLly8rNDRUnp6emjt3rnbu3KlFixZJkq5du2ZzPY6OjlbPLRaLUlNTbR43I+fOndPZs2dVtGhRSdK8efPUp08fderUSWvWrFFMTIw6duxo9bry5cunZs2aacaMGfrrr7+0cuVKqyM4nTt31vHjx/Xaa69p//79qlKlij777LNMaxgwYIASEhLMx6lTpx7KawUAAPg3YyrUYyIkJETR0dG6cOGCOcVHunlOwcqVK7Vjxw51795dknTo0CGdO3dOERER5l/Xb72qVEacnJyUkpJic50lSpRIdyL0zp0773u8Tz/9VHZ2dmrevLkkacuWLapVq5Z69Ohh9sno8rqdO3dW27ZtVahQIRUrVky1a9e2Wu7r66tu3bqpW7duGjBggL766iu99dZbGdbg7OwsZ2fn+34NAAAA4IjFYyMkJESbN29WTEyMecRCkoKDgzV58mRdu3bNPL+icOHCcnJy0meffabjx49r6dKlGjZs2B3H9/PzU1JSktavX6+///5bV65cua86u3btqkOHDql///46cuSI5s+fr8jISEm666VtL126pD///FOnTp3Sjz/+qDfeeEPDhw/XiBEjFBAQIEkKDAzUrl27tHr1ah05ckQffvhhhsEl7YjN8OHD1bFjR6tlvXv31urVq3XixAn9/PPPioqKUqlSpe7r9QIAAODeECweEyEhIfrnn38UEBCg/Pnzm+3BwcG6dOmSeVla6ea9JyIjI7VgwQKVLl1aERERGjt27B3Hr1Wrlrp166aXX35ZefPm1ejRo++rzqJFi+q7777TwoULFRQUpEmTJplXhbrbX/0HDRqkAgUKKCAgQK+99poSEhK0fv169e/f3+zTtWtXtWjRQi+//LKqV6+uc+fOWR29SGNnZ6ewsDClpKSoffv2VstSUlL05ptvqlSpUmrcuLGKFy+uL7744r5eLwAAAO6NxTAMI7uLwJNtxIgR+vLLLx/5uQmdOnXS2bNn091nw1aJiYk3T+LuPV92zo/m5oEAAAD3Kj6i6SPbVtr3ooSEBHl6et6xL+dYIMu++OILVa1aVblz59aW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/PPPm/0sFosmT56s5557Tm5ubipVqpS2bt2qY8eOqV69enJ3d1etWrUUFxdnrrN3716FhIQoR44c8vT0VOXKlbVr16471pMjRw55e3vL399f/fv3V65cubR27Vqrejt37qy8efPK09NT9evX1969e63GWLZsmapWrSoXFxflyZNHL774orksOTlZffr0UcGCBeXu7q7q1asrOjraXH7rVKgjR47IYrHo0KFDVuN/8sknKlasmPn8wIEDatKkiTw8PJQ/f3699tpr+vvvv83lly9fVvv27eXh4aECBQpo3Lhxd9wHAAAAeDAIFo+Qh4eHPDw8tHjxYiUnJ9+x77Bhw9S+fXvFxMSoZMmSeuWVV9S1a1cNGDBAu3btkmEY6tmzp9m/Xbt2KlSokHbu3Kndu3frvffek6Oj4z3VlZqaqu+//14XLlyQk5OT2d6qVSudOXNGK1eu1O7du1WpUiU1aNBA58+flyQtX75cL774op599lnt2bNH69evV7Vq1cz1e/bsqa1bt2revHnat2+fWrVqpcaNG+vo0aPpaihevLiqVKmiuXPnWrXPnTtXr7zyiqSbQad+/fqqWLGidu3apVWrVumvv/5S69atzf59+/bVxo0btWTJEq1Zs0bR0dH6+eef7/j6k5OTlZiYaPUAAABA1lgMwzCyu4j/ku+//15dunTRP//8o0qVKik4OFht2rRRUFCQ2cdiseiDDz7QsGHDJEnbtm1TzZo1NW3aNL3++uuSpHnz5qljx476559/JEmenp767LPP1KFDh3uqw8/PT6dPn5ajo6OSk5N148YN5cqVS9u3b1dAQIA2b96spk2b6syZM3J2djbXCwgIUL9+/fTGG2+oVq1a8vf315w5c9KNf/LkSfn7++vkyZPy8fEx2xs2bKhq1arpo48+UmRkpHr37q2LFy9KksaPH6+JEyfq2LFjkm4exShRooRiY2NVsmRJDR8+XJs2bdLq1avN8X777Tf5+vrq8OHD8vHxUe7cuTVnzhy1atVKknT+/HkVKlRIb7zxhsaPH5/hvhgyZIjCw8PTtfv2ni87Z7d72p9PsviIptldAgAAeEwlJibKy8tLCQkJ8vT0vGNfjlg8Yi1bttQff/yhpUuXqnHjxoqOjlalSpUUGRlp1e/WoJE/f35JUrly5azarl69av51/Z133lHnzp3VsGFDRUREWE2Tykzfvn0VExOjDRs2qHr16vrkk08UEBAg6ebUqqSkJOXOnds80uLh4aETJ06YY8fExKhBgwYZjr1//36lpKSoePHiVutv3Lgx09ratGmj+Ph4bdu2TdLNoxWVKlVSyZIlzZqioqKsxktbFhcXp7i4OF27dk3Vq1c3x8yVK5dKlChxx/0wYMAAJSQkmI9Tp07ddd8BAADAGidvZwMXFxc1atRIjRo10ocffqjOnTtr8ODBCgsLM/vcOo3JYrFk2paamirp5l/dX3nlFS1fvlwrV67U4MGDNW/ePKtzHm6XJ08eBQQEKCAgQAsWLFC5cuVUpUoVlS5dWklJSSpQoIDVORFp0s6LuNOJ10lJSbK3t9fu3btlb29vtczDwyPDdby9vVW/fn19/fXXqlGjhr7++mt1797dasxmzZpp1KhR6dYtUKCAeaQjq5ydna2OygAAACDrOGLxGChdurQuX75s8zjFixfX//73P61Zs0YtWrTQjBkz7nldX19fvfzyyxowYIAkqVKlSvrzzz/l4OBgho+0R548eSTdPKqyfv36DMerWLGiUlJSdObMmXTre3t7Z1pHu3bt9O2332rr1q06fvy42rRpYy6rVKmSfvnlF/n5+aUb093dXcWKFZOjo6O2b99urnPhwgUuaQsAAPAIECweoXPnzql+/fqaM2eO9u3bpxMnTmjBggUaPXq0Xnjhhfse959//lHPnj0VHR2tX3/9VVu2bNHOnTtVqlSpLI3z9ttva9myZdq1a5caNmyomjVrqnnz5lqzZo3i4+P1008/aeDAgebVpgYPHqxvvvlGgwcPVmxsrPbv328eTShevLjatWun9u3ba+HChTpx4oR27NihkSNHavny5ZnW0KJFC126dEndu3dXSEiI1fkZb775ps6fP6+2bdtq586diouL0+rVq9WxY0elpKTIw8NDnTp1Ut++fbVhwwYdOHBAYWFhsrPjYw4AAPCwMRXqEfLw8DDPZYiLi9P169fl6+urLl266P3337/vce3t7XXu3Dm1b99ef/31l/LkyaMWLVpkeELynZQuXVrPPPOMBg0apBUrVmjFihUaOHCgOnbsqLNnz8rb21t169Y1z/moV6+eFixYoGHDhikiIkKenp6qW7euOd6MGTM0fPhwvfvuu/r999+VJ08e1ahRQ88991ymNeTIkUPNmjXT/PnzNX36dKtl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VLJkSc2fP1/SzSMZOXPmfGDFAQAAAHgy3Few6Nixo/bu3StJeu+99/T555/LxcVF//vf/9S3b98HWiAAAACAx999nWPxv//9z/y5YcOGOnTokHbv3q2AgAAFBQU9sOIAAAAAPBnuK1jc6urVqypSpIiKFCnyIOoBMlWvXj1VqFBB48ePz+5SAAAAcJv7mgqVkpKiYcOGqWDBgvLw8NDx48clSR9++KGmTZv2QAvE4+XPP//U22+/rYCAALm4uCh//vyqXbu2Jk2apCtXrmR3eQAAAMgm9xUsRowYocjISI0ePVpOTk5me9myZTV16tQHVhweL8ePH1fFihW1Zs0affTRR9qzZ4+2bt2qfv366YcfftC6deuyu8Q7SklJUWpqanaXAQAA8K90X8Fi1qxZmjJlitq1ayd7e3uzvXz58jp06NADKw6Plx49esjBwUG7du1S69atVapUKfn7++uFF17Q8uXL1axZM0nSxYsX1blzZ+XNm1eenp6qX7++ebK/JA0ZMkQVKlTQ7Nmz5efnJy8vL7Vp00aXLl0y+1y+fFnt27eXh4eHChQooHHjxqWrJzk5WX369FHBggXl7u6u6tWrKzo62lweGRmpnDlzaunSpSpdurScnZ118uTJh7eDAAAA/sPuK1j8/vvvCggISNeempqq69ev21wUHj/nzp3TmjVr9Oabb8rd3T3DPhaLRZLUqlUrnTlzRitXrtTu3btVqVIlNWjQwLzfiSTFxcVp8eLF+uGHH/TDDz9o48aNioiIMJf37dtXGzdu1JIlS7RmzRpFR0fr559/ttpez549tXXrVs2bN0/79u1Tq1at1LhxYx09etTsc+XKFY0aNUpTp07VL7/8onz58j3I3QIAAID/775O3i5durQ2bdqU7oTt7777ThUrVnwgheHxcuzYMRmGoRIlSli158mTR1evXpUkvfnmm2rWrJl27NihM2fOyNnZWZI0duxYLV68WN99953eeOMNSTdDaGRkpHLkyCFJeu2117R+/XqNGDFCSUlJmjZtmubMmaMGDRpIkmbOnKlChQqZ2z158qRmzJihkydPysfHR5LUp08frVq1SjNmzNBHH30kSbp+/bq++OILlS9fPtPXlpycrOTkZPN5YmKiTfsKAADgv+i+gsWgQYPUoUMH/f7770pNTdXChQt1+PBhzZo1Sz/88MODrhGPsR07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375dW7duVVhYmGrXrq1GjRpluo/69u2r8ePHq3Tp0vr444/VrFkznThxQrlz5876DgcAAMA9IVg8YRwcHBQZGakuXbroyy+/VKVKlRQcHKw2bdooKChI69at044dOxQbG6vixYtLkvz9/a3GuH79ur788ksVK1ZMktSzZ08NHTrUXD527Fj1799fbdq0kSSNGjVKUVFRGj9+vD7//PN0NTk6Oio8PNx8XrRoUW3dulXz58+3Chbu7u6aOnWqOQVq4cKFcnd313PPPaccOXKoSJEiqlixotXYQUFBGjx4sCQpMDBQEydO1Pr16+8YLHr27KmWLVtKkiZNmqRVq1Zp2rRp6tevX4b9k5OTlZycbD5PTEzMdGwAAABkjKlQT6CWLVvqjz/+0NKlS9W4cWNFR0erUqVKioyMVExMjAoVKmSGioy4ubmZoUKSChQooDNnzki6+aX6jz/+UO3ata3WqV27tmJjYzMd8/PPP1flypWVN29eeXh4aMqUKTp58qRVn3LlylmdV9GoUSMVKVJE/v7+eu211zR37lxduXLFap2goCCr57fWmpmaNWuaPzs4OKhKlSp3rH3kyJHy8vIyH76+vnccHwAAAOkRLJ5QLi4uatSokT788EP99NNPCgsL0+DBg+Xq6nrXdR0dHa2eWywWGYZx37XMmzdPffr0UadOnbRmzRrFxMSoY8eOunbtmlU/d3d3q+c5cuTQzz//rG+++UYFChTQoEGDVL58eV28ePGOtaampt53rRkZMGCAEhISzMepU6ce6PgAAAD/BQSLf4nSpUvr8uXLCgoK0m+//aYjR47c1zienp7y8fFJd+naLVu2qHTp0hmus2XLFtWqVUs9evRQxYoVFRAQoLi4uHvanoODgxo2bKjRo0dr3759io+P14YNG+6r9jTbtm0zf75x44Z2796tUqVKZdrf2dlZnp6eVg8AAABkDedYPGHOnTunVq1a6fXXX1dQUJBy5MihXbt2afTo0XrhhRcUHBysunXrqmXLlvr4448VEBCgQ4cOyWKxqHHjxve0jb59+2rw4MEqVqyYKlSooBkzZigmJsY8uft2gYGBmjVrllavXq2iRYtq9uzZ2rlzp4oWLXrH7fzwww86fvy46tatq6eeekorVqxQamqqecWo+/X5558rMDBQpUqV0ieffKILFy7o9ddft2lMAAAA3BnB4gnj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
"/>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Waiting-Days-Was-the-Most-Influential-Predictor">Key Finding: Waiting Days Was the Most Influential Predictor<a class="anchor-link" href="#Key-Finding:-Waiting-Days-Was-the-Most-Influential-Predictor">¶</a></h3><p>I found that <code>waiting_days</code> had the largest permutation importance in the final Random Forest model. Shuffling this variable produced the largest decrease in ROC-AUC, indicating that the model relied heavily on the time between scheduling and the appointment when distinguishing between show and no-show appointments.</p>
<p><code>age</code> and <code>neighbourhood</code> also contributed meaningfully to model performance, while <code>sms_received</code>, <code>appointment_weekday</code>, and <code>scheduled_weekday</code> showed smaller but still noticeable predictive contributions.</p>
<p>In contrast, variables such as <code>diabetes</code>, <code>alcoholism</code>, and <code>handicap</code> had relatively little effect on ROC-AUC when permuted.</p>
<p>I interpret these results as indicators of predictive contribution rather than causal effects. In particular, the importance of <code>neighbourhood</code> may reflect underlying differences in access, scheduling, transportation, or other unobserved factors rather than location itself directly causing no-show behavior.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Neighbourhood-Sensitivity-Check">Neighbourhood Sensitivity Check<a class="anchor-link" href="#Neighbourhood-Sensitivity-Check">¶</a></h3><p>Since <code>neighbourhood</code> expands into many one-hot encoded variables, I tested whether it provided meaningful predictive value rather than interpreting the large number of neighbourhood-specific feature importances at face value.</p>
<p>I compared the Random Forest model with and without <code>neighbourhood</code> using the same training data, 3-fold cross-validation, and classification threshold of 0.24.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-python"><pre><span></span><span class="n">categorical_features_no_neighbourhood</span> <span class="o">=</span> <span class="p">[</span>
    <span class="n">feature</span>
    <span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">categorical_features</span>
    <span class="k">if</span> <span class="n">feature</span> <span class="o">!=</span> <span class="s2">"neighbourhood"</span>
<span class="p">]</span>

<span class="n">X_no_neighbourhood</span> <span class="o">=</span> <span class="n">model_df</span><span class="p">[</span>
    <span class="n">numerical_features</span> <span class="o">+</span> <span class="n">categorical_features_no_neighbourhood</span>
<span class="p">]</span>

<span class="c1"># Use the same training rows as the original model</span>
<span class="n">X_train_no_neighbourhood</span> <span class="o">=</span> <span class="n">X_no_neighbourhood</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">X_train</span><span class="o">.</span><span class="n">index</span><span class="p">]</span>

<span class="n">preprocessor_no_neighbourhood</span> <span class="o">=</span> <span class="n">ColumnTransformer</span><span class="p">(</span>
    <span class="n">transformers</span><span class="o">=</span><span class="p">[</span>
        <span class="p">(</span><span class="s2">"num"</span><span class="p">,</span> <span class="n">StandardScaler</span><span class="p">(),</span> <span class="n">numerical_features</span><span class="p">),</span>
        <span class="p">(</span>
            <span class="s2">"cat"</span><span class="p">,</span>
            <span class="n">OneHotEncoder</span><span class="p">(</span><span class="n">handle_unknown</span><span class="o">=</span><span class="s2">"ignore"</span><span class="p">),</span>
            <span class="n">categorical_features_no_neighbourhood</span>
        <span class="p">)</span>
    <span class="p">]</span>
<span class="p">)</span>

<span class="n">rf_no_neighbourhood</span> <span class="o">=</span> <span class="n">Pipeline</span><span class="p">([</span>
    <span class="p">(</span><span class="s2">"preprocessor"</span><span class="p">,</span> <span class="n">preprocessor_no_neighbourhood</span><span class="p">),</span>
    <span class="p">(</span>
        <span class="s2">"model"</span><span class="p">,</span>
        <span class="n">RandomForestClassifier</span><span class="p">(</span>
            <span class="n">n_estimators</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
            <span class="n">class_weight</span><span class="o">=</span><span class="s2">"balanced"</span><span class="p">,</span>
            <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
            <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span>
        <span class="p">)</span>
    <span class="p">)</span>
<span class="p">])</span>

<span class="n">rf_no_neighbourhood_prob</span> <span class="o">=</span> <span class="n">cross_val_predict</span><span class="p">(</span>
    <span class="n">rf_no_neighbourhood</span><span class="p">,</span>
    <span class="n">X_train_no_neighbourhood</span><span class="p">,</span>
    <span class="n">y_train</span><span class="p">,</span>
    <span class="n">cv</span><span class="o">=</span><span class="n">cv</span><span class="p">,</span>
    <span class="n">method</span><span class="o">=</span><span class="s2">"predict_proba"</span><span class="p">,</span>
    <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span>
<span class="p">)[:,</span> <span class="mi">1</span><span class="p">]</span>

<span class="n">rf_with_neighbourhood_prob</span> <span class="o">=</span> <span class="n">cv_probabilities</span><span class="p">[</span><span class="s2">"Random Forest"</span><span class="p">]</span>

<span class="n">threshold</span> <span class="o">=</span> <span class="mf">0.24</span>

<span class="n">neighbourhood_comparison</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
    <span class="s2">"Model"</span><span class="p">:</span> <span class="p">[</span>
        <span class="s2">"With Neighbourhood"</span><span class="p">,</span>
        <span class="s2">"Without Neighbourhood"</span>
    <span class="p">],</span>
    <span class="s2">"Recall"</span><span class="p">:</span> <span class="p">[</span>
        <span class="n">recall_score</span><span class="p">(</span>
            <span class="n">y_train</span><span class="p">,</span>
            <span class="p">(</span><span class="n">rf_with_neighbourhood_prob</span> <span class="o">&gt;=</span> <span class="n">threshold</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
        <span class="p">),</span>
        <span class="n">recall_score</span><span class="p">(</span>
            <span class="n">y_train</span><span class="p">,</span>
            <span class="p">(</span><span class="n">rf_no_neighbourhood_prob</span> <span class="o">&gt;=</span> <span class="n">threshold</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
        <span class="p">)</span>
    <span class="p">],</span>
    <span class="s2">"ROC-AUC"</span><span class="p">:</span> <span class="p">[</span>
        <span class="n">roc_auc_score</span><span class="p">(</span>
            <span class="n">y_train</span><span class="p">,</span>
            <span class="n">rf_with_neighbourhood_prob</span>
        <span class="p">),</span>
        <span class="n">roc_auc_score</span><span class="p">(</span>
            <span class="n">y_train</span><span class="p">,</span>
            <span class="n">rf_no_neighbourhood_prob</span>
        <span class="p">)</span>
    <span class="p">]</span>
<span class="p">})</span>

<span class="n">neighbourhood_comparison</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[ ]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div class="colab-df-container" id="df-48b4f80b-e861-4734-b053-199cf884f390">
<div>
<style scoped="">
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
<th>Model</th>
<th>Recall</th>
<th>ROC-AUC</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>With Neighbourhood</td>
<td>0.592</td>
<td>0.712</td>
</tr>
<tr>
<th>1</th>
<td>Without Neighbourhood</td>
<td>0.543</td>
<td>0.639</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">
<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-48b4f80b-e861-4734-b053-199cf884f390')" style="display:none;" title="Convert this dataframe to an interactive table.">
<svg height="24px" viewbox="0 -960 960 960" xmlns="http://www.w3.org/2000/svg">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"></path>
</svg>
</button>
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
<script>
      const buttonEl =
        document.querySelector('#df-48b4f80b-e861-4734-b053-199cf884f390 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-48b4f80b-e861-4734-b053-199cf884f390');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Key-Finding:-Neighbourhood-Added-Predictive-Value">Key Finding: Neighbourhood Added Predictive Value<a class="anchor-link" href="#Key-Finding:-Neighbourhood-Added-Predictive-Value">¶</a></h3><p>I found that removing <code>neighbourhood</code> reduced Random Forest performance. Recall decreased from 0.592 to 0.543, while ROC-AUC decreased from 0.712 to 0.639.</p>
<p>This suggests that neighbourhood information contributed meaningful predictive information rather than appearing prominently only because it was represented by many one-hot encoded variables.</p>
<p>Based on this comparison, I retained <code>neighbourhood</code> in the final model. However, I interpret neighbourhood-related patterns cautiously because they may reflect underlying differences in access, scheduling, transportation, or other unobserved factors rather than direct causal effects of location itself.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="10.-Conclusion">10. Conclusion<a class="anchor-link" href="#10.-Conclusion">¶</a></h2><p>In this project, I went beyond exploratory data analysis by framing medical appointment attendance as a binary classification problem and building machine learning models to predict patient no-shows.</p>
<p>My exploratory analysis identified several patterns associated with appointment attendance, particularly the relationship between longer waiting periods and higher no-show rates. I also found that the apparent relationship between SMS reminders and no-shows changed when appointments were compared within similar waiting periods, highlighting the importance of considering other factors when interpreting observational data.</p>
<p>I compared Logistic Regression and Random Forest using stratified 3-fold cross-validation. Although Logistic Regression initially achieved higher recall at the default 0.5 threshold, Random Forest achieved a higher ROC-AUC. When I compared the models at a similar recall level, Random Forest achieved slightly higher precision. Based on this trade-off, I selected Random Forest with a classification threshold of 0.24 as the final model.</p>
<p>On the holdout test set, the final model correctly identified 2,613 of 4,463 actual no-shows, corresponding to approximately 58.5% recall. The model achieved an ROC-AUC of approximately 0.72, indicating moderate ability to distinguish between show and no-show appointments.</p>
<p>I also tested whether <code>neighbourhood</code> materially contributed to prediction performance. Removing it reduced recall from 0.592 to 0.543 and ROC-AUC from 0.712 to 0.639, so I retained it in the final model while interpreting geographic patterns cautiously.</p>
<p>Overall, this project demonstrates how SQL, Python, feature engineering, and machine learning can be combined to move from exploratory analysis to a practical no-show risk prediction workflow.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Limitations-and-Next-Steps">Limitations and Next Steps<a class="anchor-link" href="#Limitations-and-Next-Steps">¶</a></h3><p>I observed a relatively high number of false-positive predictions in the final model, so I interpret its predictions as risk indicators rather than definitive no-show outcomes. I also recognize that the dataset is limited to the available appointment and patient variables, and I do not interpret the observed relationships, including neighbourhood-related patterns, as causal effects. Moreover, I observed differences in recall across age groups and scholarship status, suggesting that the model may not identify no-shows equally well across all patient groups.</p>
<p>Because some patients had multiple appointment records, I recognize that the appointment-level random split may have placed records from the same patient in both the training and test sets. In future work, I could use a patient-level grouped split to prevent this overlap and provide a more rigorous evaluation on previously unseen patients.</p>
<p>I could also incorporate additional information such as prior appointment history, transportation or access-related factors, and more detailed scheduling behavior. In addition, I could evaluate alternative classification thresholds, additional feature engineering approaches, and model tuning strategies to improve the balance between identifying no-shows and reducing false-positive predictions.</p>
</div>
</div>
</div>
</div>
</div>
