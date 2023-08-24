from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import numpy as np


def assembler(df):
    va = VectorAssembler(
        inputCols=["sess_authorised_percent", "sess_unauthorised_percent"],
        outputCol="features")
    return va.transform(df)


def train_model(df):
    df_absent_rate = df.select(df.sess_authorised_percent, df.sess_unauthorised_percent, df.sess_overall_percent) \
        .filter((df.geographic_level == 'Local authority') & (df.school_type == 'Total'))

    df_absent_rate_va = assembler(df_absent_rate)

    train_data, test_data = df_absent_rate_va.randomSplit([0.8, 0.2], seed=42)

    lr = (LinearRegression(featuresCol='features', labelCol="sess_overall_percent", predictionCol='calc',
                           maxIter=10, regParam=0.3, elasticNetParam=0, standardization=False))

    model = lr.fit(train_data)
    model.transform(test_data)
    return model


def predict(df, model):
    df_absent_rate = df.filter((df.geographic_level == 'Local authority') & (df.school_type == 'Total')) \
        .groupBy(df.la_name) \
        .agg({'sess_authorised_percent': 'avg', 'sess_unauthorised_percent': 'avg'}) \
        .withColumnRenamed('avg(sess_authorised_percent)', 'sess_authorised_percent') \
        .withColumnRenamed('avg(sess_unauthorised_percent)', 'sess_unauthorised_percent')

    df_absent_rate_va = assembler(df_absent_rate)
    predictions = model.transform(df_absent_rate_va)
    df_pred = predictions.drop('features').collect()
    r1 = np.core.records.fromrecords(df_pred, names=tuple(df_pred[0].asDict()))
    print()
    print('Region prediction with the best pupil attendance is: ', r1.la_name[np.argmin(r1.calc)], end='\n\n')
