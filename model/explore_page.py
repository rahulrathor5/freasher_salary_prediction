import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

import altair as alt
import plotly.express as px
import plotly.figure_factory as ff

@st.cache 
def load_data():
    df = pd.read_csv("data.csv")
    df['salary'] = df['salary'].replace(to_replace = np.nan, value= 0.0)
    df.drop(['sl_no'],axis = 1, inplace=True)
    return df

df = load_data()
 
def show_explore_page():
    st.title("Explore Fresher Salaries")

    # chart = alt.Chart(df).mark_bar().encode(
    # alt.X("status"),
    # y='count()')
    # st.altair_chart(chart)
    st.header("Distribution of Salary [0:Means Not Placed Students]")
    fig_0 = px.histogram(df, x="salary")
    st.plotly_chart(fig_0, use_container_width=True)

    # st.header("Distribution of Salary")
    # sns.distplot(df['salary'])
    # st.pyplot()

    st.header("Frequency of Specialisation with respect to Status")

    # Set the figure
    plt.figure(figsize=(10, 6))

     # Use string column names, not Series
    sns.countplot(data=df, x='specialisation', hue='status')

    # Display plot in Streamlit
    st.pyplot(plt)

    st.header('Bar Plot to shows Specialisation and Salary Relationship')
    fig = px.bar(data_frame=df, x="specialisation",y="salary")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Relationship between SSC Per and HSC Per")
    fig1 = px.scatter(df, x="ssc_p", y="hsc_p") 
    st.plotly_chart(fig1)

    st.header(" Shows which Degree Type makes more Salary")
    fig2 = px.bar(data_frame=df, x="degree_t",y="salary")
    st.plotly_chart(fig2, use_container_width=True)

    st.header("HSC Subjects Impacts on Salary")
    fig3 = px.box(df, x="hsc_s", y="salary") 
    st.plotly_chart(fig3, use_container_width=True)

    st.header("MBA Percentage relationship with Salary")
    fig4 = px.scatter(df, x="mba_p", y="salary") 
    plt.show()
    st.plotly_chart(fig4, use_container_width=True)


    # fig5 = px.histogram(df, x="salary", y="mba_p", color="specialisation", marginal="rug",hover_data=df.columns)
    # st.plotly_chart(fig5, use_container_width=True)

    fig = sns.catplot(x='status', data=df, kind='count', height=5, aspect=1.2)

  # Set the title using the underlying matplotlib axes
    fig.ax.set_title("Frequency of Status (Placed or Not Placed)")

# Draw with Streamlit
    st.pyplot(fig.fig)

    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x='specialisation', data=df)
    plt.title('Frequency of Specialisation')
    st.pyplot(fig1)
    
    # Plot 2: Regplot (Salary vs Degree Percentage)
    fig2 = plt.figure(figsize=(10, 5))
    sns.regplot(x='degree_p', y='salary', data=df)
    plt.title("Relationship Between Salary and Degree %")
    st.pyplot(fig2)
    
    # Plot 3: Heatmap for Correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Plot heatmap for numeric features
    fig3 = plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Among Numeric Columns')
    st.pyplot(fig3)
    

    # with st.echo(code_location='below'):

   

    #     fig = plt.figure()
    #     ax = fig.add_subplot(1,1,1)

    #     ax.scatter(
    #         df["salary"],
    #         df["etest_p"],
    #     )

    #     ax.set_xlabel("Salary")
    #     ax.set_ylabel("Entrance Test Percentage")

    #     st.write(fig)
    