
from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

# Import necessary scikit-learn modules for ML
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/visualize", methods=["POST"])
def visualize():
        file = request.files["csv_file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            columns = df.columns.tolist()

            return render_template("visualize.html" , columns=columns, filename=file.filename)
        
@app.route("/plot", methods=["POST"])
def plot():
    column = request.form["column"]
    filename = request.form["filename"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"],filename)
    df = pd.read_csv(filepath)
    
    df = df.sample(n=min(300, len(df)), random_state=1)
    column_data = df[column]

    plt.figure(figsize=(10, 6))
    summary = ""

    if pd.api.types.is_numeric_dtype(column_data):
        sns.histplot(column_data.dropna(),kde=True)
        chart_title = f"Histogram of {column}"

        summary={
            "Type": "Numeric",
            "Mean": column_data.mean(),
            "Median": column_data.median(),
            "Stander Deviation": column_data.std(),
            "MIN": column_data.min(),
            "MAX": column_data.max()
        }

    else:
        top_categories = column_data.value_counts().index[:10]
        df = df[df[column].isin(top_categories)]
        sns.countplot(x=column,data=df)
        plt.xticks(rotation=45)
        chart_title = f"Bar Chart of {column}"

        summary={
            "Type": "Categorical",
            "Unique value":column_data.unique(),
            "Most frequent": column_data.mode()[0],
            "Top 3 Categories": column_data.value_counts().head(3).to_dict()
        }

    plt.title(chart_title)
    plt.tight_layout()

    chart_path = os.path.join("static","column_chart.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template("show_chart.html", chart=chart_path, column=column, summary=summary)


@app.route("/plot2", methods=["POST"])
def plot2():
    column1 = request.form["column1"]
    column2 = request.form["column2"]
    filename = request.form["filename"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"],filename)
    df = pd.read_csv(filepath)

    df = df.sample(n=min(300,len(df)),random_state=1)

    if pd.api.types.is_object_dtype(df[column1]):
        top_x = df[column1].value_counts().index[:10]
        df = df[df[column1].isin(top_x)]
    if pd.api.types.is_object_dtype(df[column2]):
        top_y = df[column2].value_counts().index[:10]
        df = df[df[column2].isin(top_y)]

    x = df[column1]
    y = df[column2]

    plt.figure(figsize=(10, 6))
    summary=""

    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        sns.scatterplot(x=x, y=y)
        chart_title = f"Scatterplot of {column1} & {column2}"

        summary={
            "Type":"Numerical vs Numerical",
            f"Mean : {column1}":x.mean(),
            f"Mean : {column2}":y.mean()
        }
    
    elif pd.api.types.is_numeric_dtype(x) and pd.api.types.is_object_dtype(y):
        sns.boxplot(x=y, y=x)
        chart_title = f"Boxplot of {column1} & {column2}"

        summary={
            "Type":"Numerical vs Categorical",
            f"Mean : {column1}":x.mean(),
            "Top 3 Categories": y.value_counts().head(3).to_dict()
        }
    
    elif pd.api.types.is_numeric_dtype(y) and pd.api.types.is_object_dtype(x):
        sns.boxplot(x=x, y=y)
        chart_title = f"Boxplot of {column1} & {column2}"

        summary={
            "Type":"Categorical vs Numerical",
            f"Mean : {column2}":y.mean(),
            "Top 3 Categories": x.value_counts().head(3).to_dict()
        }

    else:
        cross_tab = pd.crosstab(x, y)
        sns.heatmap(cross_tab, annot=True, cmap="Blues")
        chart_title = f"Heatmap of {column1} & {column2}"

        summary = {
        "Type": "Categorical vs Categorical",
        f"Unique: {column1}": x.nunique(),
        f"Unique: {column2}": y.nunique()
    }


    plt.title(chart_title)
    plt.tight_layout()

    chart_path = os.path.join("static","column_chart.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template("show_chart.html", chart=chart_path, column=[column1,column2], summary=summary)


@app.route('/ml_analysis')
def ml_analysis():
    
    filename = request.args.get('filename')
    if not filename:
        return "No filename provided for ML analysis.", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        df = pd.read_csv(filepath)
        columns = df.columns.tolist() # Get columns for selection
    except FileNotFoundError:
        return "File not found. Please upload it again.", 404
    except Exception as e:
        return f"Error reading CSV file: {e}", 400

    return render_template("ml_analysis.html", filename=filename, columns=columns)

@app.route('/ml_insights', methods=["POST"])
def ml_insights():

    filename = request.form["filename"]
    model_type = request.form["model"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return "File not found. Please upload it again.", 404
    except Exception as e:
        return f"Error reading CSV file: {e}", 400

    result_plot = None
    result_table = None

    
    # Select only numeric columns for KMeans, drop NaNs for simplicity
    numeric_data = df.select_dtypes(include="number").dropna()

    if numeric_data.shape[1] < 2:
        return "Not enough numeric columns (at least 2) for KMeans clustering after dropping NaNs.", 400
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    if model_type == "kmeans":
        n_cluster = int(request.form.get("cluster", 3))
        # Ensure n_clusters is not greater than the number of samples
        if n_cluster > len(scaled_data):
            return f"Number of clusters ({n_cluster}) cannot be greater than the number of samples ({len(scaled_data)}). Please reduce the cluster count.", 400

        kmeans_model = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto') # n_init='auto' to avoid future warning
        labels = kmeans_model.fit_predict(scaled_data)
        numeric_data["Cluster"] = labels

        #summarize the data for better understanding
        summary = numeric_data.groupby("Cluster").mean().round(2)
        counts = numeric_data["Cluster"].value_counts().sort_index()
        summary["Count"]=counts
        summary = summary.reset_index()
        

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        plot_data = pd.DataFrame({
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "Cluster": labels
        })

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="PCA1",
            y="PCA2",
            hue=plot_data["Cluster"],
            palette="Set2",
            data=plot_data,
            legend="full"
        )

        plt.title(f"KMeans Clustering (PCA View) with k={n_cluster}")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        result_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        result_table = numeric_data.to_html(classes="table table-striped", index=False)
        result_table2 = summary.to_html(classes="table table-striped", index=False)

    elif model_type == "dbscan":
        eps = float(request.form.get("eps"))
        min_samples = int(request.form.get("min_samples"))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_data)

        numeric_data["Cluster"] = labels

        summary = numeric_data.groupby("Cluster").mean().round(2)
        counts = numeric_data["Cluster"].value_counts().sort_index()
        summary["Count"]=counts
        summary = summary.reset_index()

        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        plot_data = pd.DataFrame({
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "Cluster": labels
        })

        plt.figure(figsize=(10, 6))
        unique_labels = sorted(plot_data["Cluster"].unique())

        # Create custom palette: gray for noise
        base_palette = sns.color_palette("Set2", len([l for l in unique_labels if l != -1]))
        full_palette = [(0.5, 0.5, 0.5)] + base_palette if -1 in unique_labels else base_palette
        label_to_color = {label: full_palette[i] for i, label in enumerate(unique_labels)}

        sns.scatterplot(
            x="PCA1",
            y="PCA2",
            hue=plot_data["Cluster"],
            palette=label_to_color,
            data=plot_data,
            legend="full"
        )

        plt.title(f"DBSCAN Clustering (PCA View) (eps={eps}, min_samples={min_samples})")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        result_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        result_table = numeric_data.to_html(classes="table table-striped", index=False)
        result_table2 = summary.to_html(classes="table table-striped", index=False)


    else:
        return "Invalid model type selected.", 400

    return render_template(
        "ml_result.html",
        plot=result_plot,
        table=result_table,
        table2=result_table2
    )


@app.route('/advance_analysis')
def advance_analysis():
    
    filename = request.args.get('filename')
    if not filename:
        return "No filename provided for ML analysis.", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        df = pd.read_csv(filepath)
        columns = df.columns.tolist() # Get columns for selection
    except FileNotFoundError:
        return "File not found. Please upload it again.", 404
    except Exception as e:
        return f"Error reading CSV file: {e}", 400

    return render_template("advance_analysis.html", filename=filename, columns=columns)

@app.route("/feature_importance", methods=["POST"])
def feature_importance():
    filename = request.form["filename"]
    target = request.form["target"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"],filename)

    df = pd.read_csv(filepath)
    df_encoded = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df_encoded[col] = pd.factorize(df[col])[0]

    if target not in df_encoded:
        return "Target column not found in the table."

    x = df_encoded.drop(columns=target)
    y = df_encoded[target]

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        model = ExtraTreesRegressor()
    else:
        model = ExtraTreesClassifier()
    model.fit(x,y)

    importance = model.feature_importances_
    feature_name = x.columns

    imp_arr = np.array(importance)
    max_index = np.argmax(imp_arr)
    min_index = np.argmin(imp_arr)
    max_feature = feature_name[max_index]
    min_feature = feature_name[min_index]

    summary=""
    summary={
            "Name": "Importance Feature Visualization",
            "Most important feature": max_feature,
            "Least important feature":min_feature
        }

    plt.figure(figsize=(10,6))
    sns.barplot(x=importance, y=feature_name)
    plt.title("Feature Importance")
    plt.tight_layout()

    chart_path = os.path.join("static", "feature_importance.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template("show_chart.html",column=target, chart=chart_path, summary=summary)

@app.route("/anomaly")
def anomaly():
    
    filename = request.args.get('filename')
    if not filename:
        return "No filename provided for ML analysis.", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return "File not found. Please upload it again.", 404
    except Exception as e:
        return f"Error reading CSV file: {e}", 400

    df_encoded = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df_encoded[col] = pd.factorize(df[col])[0]
    
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df_encoded)
    df["Anomaly"] = model.predict(df_encoded)

    anomalies = df[df["Anomaly"] == -1]
    normal = df[df["Anomaly"] == 1]

    plt.figure(figsize=(10,6))
    plt.scatter(normal.iloc[:,0], normal.iloc[:,1], label="Normal", alpha=0.6)
    plt.scatter(anomalies.iloc[:,0], anomalies.iloc[:,1], label="Anomaly",color="red")
    plt.legend()
    plt.title("Anomaly Detection (Red = Outliers)")
    plt.tight_layout()

    chart_path = os.path.join("static", "anomaly_chart.png")
    plt.savefig(chart_path)
    plt.close()

    anomalies = anomalies.drop(columns=["Anomaly"])
    anomalies_table = anomalies.to_html(classes='data',index=False)
    normal = normal.drop(columns=["Anomaly"])
    normal_table = normal.to_html(classes='data',index=False)

    return render_template("anomaly_show.html",chart=chart_path,table=anomalies_table,table2=normal_table)

