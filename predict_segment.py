from flask import Flask, request, jsonify, send_file
import io
import os

import pandas as pd
import datetime as dt
from datetime import timedelta

app = Flask("predict-segment")

seg_mapping = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions",
}

rec_bins = [1.0, 13.8, 33.0, 72.0, 180.0, 374.0]
freq_bins = [1.0000e00, 8.6840e02, 1.7358e03, 2.6032e03, 3.4706e03, 4.3380e03]
mon_bins = [3.7500000e00, 2.5019400e02, 4.9009600e02, 9.4227600e02, 2.0584260e03, 2.8020602e05]


def process_df(df):
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    max_date = df["InvoiceDate"].max()

    today_date = max_date + timedelta(days=2)
    rfm = df.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda invoice_date: (today_date - invoice_date.max()).days,
            "InvoiceNo": lambda invoice: invoice.nunique(),
            "TotalPrice": lambda total_price: total_price.sum(),
        }
    )

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[rfm["monetary"] > 0]
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm.reset_index()
    return rfm


def generate_scores(rfm_df):
    # recency_score
    rfm_df["recency_score"] = pd.cut(rfm_df["recency"], bins=rec_bins, labels=[5, 4, 3, 2, 1])
    # frequency_score
    rfm_df["frequency_score"] = pd.cut(
        rfm_df["frequency"].rank(method="first"), bins=freq_bins, labels=[1, 2, 3, 4, 5]
    )
    # monetary_score
    rfm_df["monetary_score"] = pd.cut(rfm_df["monetary"], bins=mon_bins, labels=[1, 2, 3, 4, 5])

    rfm_df["RFM_SCORE"] = rfm_df["frequency_score"].astype(str) + rfm_df["monetary_score"].astype(str)

    return rfm_df


def map_segments(rfm_df):
    rfm_df["segment"] = rfm_df["RFM_SCORE"].replace(seg_mapping, regex=True)
    return rfm_df


def pipeline(df):
    rfm_df = process_df(df)
    rfm_df = generate_scores(rfm_df)
    rfm_df = map_segments(rfm_df)
    return rfm_df


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # customer_data = request.get_json()

    if request.files:
        uploaded_file = request.files["file"]
        filepath = os.path.join(app.config["FILE_UPLOADS"], uploaded_file.filename)
        uploaded_file.save(filepath)

        with open(filepath) as file:
            customer_data = pd.read_csv(file)
            rfm_df = pipeline(customer_data)
            buffer = io.StringIO()
            rfm_df.to_csv(buffer)
            buffer.seek(0)

            mem = io.BytesIO()
            mem.write(buffer.getvalue().encode())
            mem.seek(0)
            buffer.close()

            return send_file(mem, mimetype="text/csv")


if __name__ == "__main__":
    app.config["FILE_UPLOADS"] = "./files/"
    app.run(debug=True, host="0.0.0.0", port=9696)

    # # Testing function
    # df = pd.read_csv("./small_data.csv")
    # print(map_segments(generate_scores(process_df(df))))
