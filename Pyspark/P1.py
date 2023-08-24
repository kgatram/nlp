"""

"""
from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import numpy as np
import plotly.express as px
import pkg_resources
import P1ML

global spark
global df


def create_spark_session():
    global spark
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("P1") \
        .getOrCreate()

    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.sparkContext.setLogLevel("ERROR")


def load_data():
    global df
    path = "Absence_3term201819_nat_reg_la_sch.csv"
    try:
        df_load = spark.read.load(path, format="csv", sep=",", inferSchema="true", header="true")
    except Exception:
        print("Please check the path to the data file.")
        print("Application terminated.")
        if spark is not None:
            spark.stop()
        exit(1)
    else:
        df = df_load.persist()


def function1():
    """
        Given a list of local authorities, display in a well-formatted fashion
        the number of pupil enrolments in each local authority by time period.
    """

    lauth = input('Enter the list of local authorities (eg. Camden,Greenwich): ')
    la = lauth.split(',')
    for i in range(len(la)):
        lai = la[i].strip().lower()
        local_authority = df.select(df.la_name, df.time_period, df.enrolments) \
            .filter(f.lower(df.la_name).isin(lai) & (f.lower(df.geographic_level) == 'local authority') &
                    (f.lower(df.school_type) == 'total')) \
            .orderBy(df.la_name, df.time_period)
        local_authority.show(truncate=False)


def function2():
    """
    Allow the user to search the dataset by school type, showing the total
    number of pupils who were given authorised absences because of medical
    appointments or illness in the time period 2017-2018.
    :return:
    """

    type = input('Enter the school type (eg. State-funded primary): ').lower()
    year = 201718
    df.select(df.school_type, df.time_period, df.sess_auth_appointments.alias('apoint'),
              df.sess_auth_illness.alias('illness')) \
        .withColumn('auth_med_ill', f.col('apoint') + f.col('illness')) \
        .filter((df.time_period == year) & (f.lower(df.school_type) == type) & (df.geographic_level == 'National')) \
        .show(truncate=False)


def function3():
    """
    Allow a user to search for all unauthorised absences in a certain year,
    broken down by either region name or local authority name.
    """
    try:
        level = input('Enter level (eg. regional or local authority) :').lower().strip()
        year = input('Enter year eg 20xx :').strip()
        year1 = int(year) + 1
        year = int(year + str(year1)[2:])
        order = 'region_name' if 'regional' in level else 'la_name'
    except ValueError:
        print('Invalid input')
    else:
        output = df.select(df.region_name, f.coalesce(df.la_name, f.lit("*All")).alias('Local Authority'),
                           df.time_period,
                           df.sess_unauthorised.alias('unauth')) \
            .filter((df.time_period == year) & (df.school_type == 'Total') & (f.lower(df.geographic_level) == level)) \
            .orderBy(order)

        output.show(truncate=False)
        if output.count() > 20:
            show_all = input('Show all? (y/n) :').lower()
            if show_all == 'y':
                li = output.collect()
                print()
                print('-' * 100)
                for i in range(len(li)):
                    print(li[i].asDict())

                print('-' * 100)
        print()


def function4():
    """
    List the top 3 reasons for authorised absences in each year.
    """

    def top3(rec):
        rec_dict = dict()
        rec_dict.update(rec.asDict())
        a = np.array(list(rec_dict.values()))
        b = np.argsort(a)[::-1]
        print(a[0], list(rec_dict)[b[0]], list(rec_dict)[b[1]], list(rec_dict)[b[2]])

    df.select(df.time_period, df.sess_auth_appointments.alias('medical'), df.sess_auth_excluded.alias('excluded'),
              df.sess_auth_illness.alias('illness'), df.sess_auth_holiday.alias('holidays'),
              df.sess_auth_other.alias('others'),
              df.sess_auth_religious.alias('religious'), df.sess_auth_study.alias('study'),
              df.sess_auth_traveller.alias('travel')) \
        .filter((df.geographic_level == 'National') & (df.school_type == 'Total')) \
        .orderBy(df.time_period) \
        .withColumnRenamed("sum(excluded)", "excluded") \
        .withColumnRenamed("sum(medical)", "medical") \
        .withColumnRenamed("sum(holidays)", "holidays") \
        .withColumnRenamed("sum(illness)", "illness") \
        .withColumnRenamed("sum(others)", "others") \
        .withColumnRenamed("sum(religious)", "religious") \
        .withColumnRenamed("sum(study)", "study") \
        .withColumnRenamed("sum(travel)", "travel") \
        .foreach(top3)

    print()


def function5():
    """
    Allow a user to compare two local authorities of their choosing in a given year
    :return:
    """
    try:

        lauth = input('Enter the two local authorities (eg. Camden,Greenwich) to compare: ').lower()
        authority = lauth.split(',')
        authority = [i.strip() for i in authority]
        year = input('Enter year eg 20xx :').strip()
        year1 = int(year) + 1
        period = int(year + str(year1)[2:])

    except ValueError:
        print('Invalid input', end='\n\n')
        return

    compare_data = df.select(df.la_name, df.time_period, df.enrolments, df.num_schools,
                             df.sess_auth_holiday.alias('auth_holiday'),
                             df.sess_unauth_holiday.alias('unauth_holiday')) \
        .filter((f.lower(df.la_name).isin(authority))
                & (df.time_period == period)
                & (df.geographic_level == 'Local authority')
                & (df.school_type == 'Total')) \
        .orderBy(df.la_name) \
        .collect()

    if len(compare_data) == 0:
        print('No data found', end='\n\n')
        return

    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    rec_dict = dict()

    for rec in compare_data:
        rec_dict = rec.asDict()
        offset = width * multiplier
        rects = ax.bar(x + offset, list(rec_dict.values())[-2:], width, label=list(rec_dict.values())[0])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of holiday sessions')
    ax.set_title('Authorised Vs Unauthorised Holidays')
    ax.set_xticks(x, list(rec_dict)[-2:])
    ax.legend(loc='best', ncols=2)
    plt.suptitle('Year ' + '-'.join([str(list(rec_dict.values())[1])[:-2], str(list(rec_dict.values())[1])[-2:]]))

    plt.show()
    print()


def function6():
    """
    Chart/explore the performance of regions in England from 2006-2018.
    Your charts and subsequent analysis in your report should answer the
    following questions: Are there any regions that have improved in pupil
    attendance over the years? Are there any regions that have worsened?
    Which is the overall best/worst region for pupil attendance?
    :return:
    """

    recs = df.select(df.region_name, df.time_period, df.sess_overall_percent) \
        .filter((df.geographic_level == 'Regional') & (df.school_type == 'Total')) \
        .orderBy(df.region_name, df.time_period) \
        .collect()

    if len(recs) == 0:
        print('No data found', end='\n\n')
        return

    r = np.core.records.fromrecords(recs, names=tuple(recs[0].asDict()))
    fig, ax = plt.subplots(layout='constrained')
    x = np.unique(r.time_period)
    l = np.unique(r.region_name)

    for i in range(0, len(r.sess_overall_percent), 13):
        y = r.sess_overall_percent[i:i + 13]
        ax.plot(x, y, label=l[i // 13], marker='|')

    ax.set_xlabel('Years')
    ax.set_ylabel('Overall absence rate %')
    ax.set_title('Performance of regions in England')
    ax.legend(loc='best')
    plt.suptitle('Period: 2006-2018')
    plt.show()
    print()


def function7():
    """
    Visualisation and interaction of the dataset, using external libraries.
    :return:
    """

    if 'plotly' not in [p.project_name for p in pkg_resources.working_set]:
        print('Plotly not found.', end='\n\n')
        return

    df_unauth = df.select(df.region_name, df.time_period, df.sess_unauthorised_percent) \
        .filter((df.geographic_level == 'Regional') & (df.school_type == 'Total')) \
        .orderBy(df.time_period) \
        .collect()

    r1 = np.core.records.fromrecords(df_unauth, names=tuple(df_unauth[0].asDict()))

    fig = px.scatter(y=r1.region_name, x=r1.sess_unauthorised_percent,
                     hover_name=r1.time_period, title='UnAuthorised Absent Rate %',
                     animation_frame=r1.time_period, orientation='h',
                     labels={'x': 'Unauthorised Absent rate %', 'y': '', 'animation_frame': 'time period'})
    fig.show()


def function8():
    """
    Analyse whether there is a link between school type, pupil absences and
    the location of the school. For example, is it more likely that schools of
    type X will have more pupil absences in location Y?
    :return:
    """

    for typ in ['State-funded primary', 'State-funded secondary', 'Special']:
        table1 = df.select(df.school_type, df.la_name, df.region_name, df.time_period,
                           df.sess_overall_percent.alias('Absent %')) \
            .filter((df.geographic_level == 'Local authority') & (df.school_type == typ)) \
            .orderBy(df.sess_overall_percent.desc()) \
            .limit(10)

        table1.show(truncate=False)

        row1 = table1.groupBy('la_name', 'region_name') \
            .agg({'la_name': 'count'}) \
            .select(f.max_by('la_name', 'count(la_name)'), f.max_by('region_name', 'count(la_name)'),
                    f.max('count(la_name)')) \
            .collect()

        print('Absent % is frequently high in {} and {} region.'.format(row1[0][0], row1[0][1]), end='\n\n')


def function9():
    """
    Predict schools with the best pupil attendance.
    :return:
    """
    P1ML.predict(df, P1ML.train_model(df))


def quitt():
    if spark is not None:
        spark.stop()


functions = {'1': function1,
             '2': function2,
             '3': function3,
             '4': function4,
             '5': function5,
             '6': function6,
             '7': function7,
             '8': function8,
             '9': function9,
             'Q': quitt}


def main():
    choice = ' '
    while choice.upper() != 'Q':
        print('Choose from the following options:')
        print('1. Show number of pupil enrolments in a local authority by time period.')
        print('2. Show authorised absences because of medical appointments or illness in 2017-18.')
        print('3. Show unauthorised absences in a year either by region or local authority.')
        print('4. List the top 3 reasons for authorised absences in each year.')
        print('5. Compare two local authorities in a given year.')
        print('6. Explore the performance of regions in England from 2006-2018.')
        print('7. Visualisation and interaction (uses Plotly).')
        print('8. Analyse school type, pupil absences and the location of the school.')
        print('9. Predict region with best pupil attendance.')
        print('Q. Quit')
        choice = input('Enter your choice: ')

        try:
            func = functions[choice.upper()]
        except KeyError:
            print('Invalid choice', end='\n\n')
        else:
            if spark is None and choice.upper() != 'Q':
                create_spark_session()
                load_data()
            func()


if __name__ == '__main__':
    spark = None
    main()
