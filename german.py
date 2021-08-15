import streamlit as st
import pandas as pd
import joblib
import altair as alt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral9
from bokeh.transform import factor_cmap

def get_data():
    df = pd.read_csv('data/articles.csv', header = None, sep = ';', quotechar = "'", names = ['label', 'text'])
    df['total_words'] = [len(x.split()) for x in df['text'].tolist()]
    return df

def bokeh_plot(df):
    group = df.groupby('label')
    source = ColumnDataSource(data = group)

    index_cmap = factor_cmap('label', palette = Spectral9, factors = sorted(df['label'].unique()), end = 1)

    p = figure(plot_width = 800, plot_height = 300, title = "Number of articles labled",
            x_range = group, toolbar_location = None, tooltips = [ ("Articles Count", "@total_words_count"),
                                                                ("Average Words in 1 article", "@total_words_mean{0}"),
                                                                ("Max Words in 1 article", "@total_words_max"),
                                                                ("Min Words in 1 article", "@total_words_min") ])

    p.vbar(x = 'label', top = 'total_words_count', width = 1, source = source,
        line_color = "white", fill_color = index_cmap)

    p.y_range.start = 0
    p.x_range.range_padding = 0.05
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Labels Frequencies"
    p.xaxis.major_label_orientation = 1.2
    p.outline_line_color = None
    return p

def predict_label(text, pipeline):
    result = pipeline.predict([text])
    return result[0]

def get_probability(text, pipeline):
    result = pipeline.predict_proba([text])
    return result

def main():
    st.title('10kGNAD Overview and Classification')

    #getting the data from csv to a dataframe
    df_articles = get_data()

    #fancy bokeh plot
    if st.checkbox('Show 10kGNAD Data Chart'):
        p = bokeh_plot(df_articles)
        st.bokeh_chart(p, use_container_width=True)

    #for different models showing the reports
    model_choice = ['Naive Bayes', 'SVM', 'Log Reg']
    choice = st.selectbox("Choose the Model", model_choice)

    if choice == 'Naive Bayes':
        pipeline = joblib.load(open('models/naive_bayes_classifier.sav', 'rb'))
        if st.checkbox('Show model report'):
            df = pd.read_csv('reports/naive_bayes_report.csv', index_col = 0)
            st.dataframe(df.style.format({"precision" : "{:.2%}", "recall" : "{:.2%}", "f1-score" : "{:.2%}", "support" : "{:.5}"}))

    elif choice == 'SVM':
        pipeline = joblib.load(open('models/svc_classifier.sav', 'rb'))
        if st.checkbox('Show model report'):
            df = pd.read_csv('reports/svc_report.csv', index_col = 0)
            st.dataframe(df.style.format({"precision" : "{:.2%}", "recall" : "{:.2%}", "f1-score" : "{:.2%}", "support" : "{:.5}"}))

    else :
        pipeline = joblib.load(open('models/log_reg_classifier.sav', 'rb'))
        if st.checkbox('Show model report'):
            df = pd.read_csv('reports/log_reg_report.csv', index_col = 0)
            st.dataframe(df.style.format({"precision" : "{:.2%}", "recall" : "{:.2%}", "f1-score" : "{:.2%}", "support" : "{:.5}"}))

    #enter the text to process
    with st.form(key = 'form'):
        text = st.text_area("")
        submit_text = st.form_submit_button(label = 'Submit')

    #process the text and display the
    if submit_text:
        col1, col2 = st.beta_columns(2)

        prediction = predict_label(text, pipeline)
        probability = get_probability(text, pipeline)

        with col1:
            st.info('Original Text')
            st.write(text)

            st.success('Predicted Label')
            st.write(prediction)

        with col2:
            st.info('Prediction Probability')
            probability_df = pd.DataFrame(probability, columns = pipeline.classes_)
            probability_renamed = probability_df.T.copy()
            probability_renamed.columns = ["probability"]
            st.dataframe(probability_renamed.sort_values(by = ["probability"], ascending = False).style.format("{:.2%}"))

            probability_df_new = probability_df.T.reset_index()
            probability_df_new.columns = ["label","probability"]
            p = alt.Chart(probability_df_new).mark_bar().encode(
                    alt.X('label',sort = alt.EncodingSortField(field = "probability", order = 'descending')),
                    alt.Y('probability'),
                    color = 'label')
            st.altair_chart(p,use_container_width = True)

if __name__ == '__main__':
    main()