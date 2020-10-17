# import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, cast, Integer
from flask import Flask, render_template, redirect, jsonify, request
import pandas as pd
import os



app = Flask(__name__)

# database setup
# Configure settings for RDS
rds_connection_string = "postgres:postgres@mypostgresaws2020.c002u923zftw.us-east-2.rds.amazonaws.com:5432/FinalProjectGroup3"
engine = create_engine(f'postgresql://{rds_connection_string}')

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)
# print(Base.classes.keys())
# Save reference to the table
Testfinal = Base.classes.testfinal
Trainfinal = Base.classes.trainfinal


@app.route("/")
def index():

    # Find one record of data from the mongo database
    # data = mongo.db.data.find()

    # Return template and data
    return render_template("index.html")



@app.route("/testfinal")
def testfinal():

    session = Session(engine)
    results = []
    result = session.execute(Testfinal.__table__.select())
    for row in result:
        results.append(dict(row))
    session.close()

    return jsonify(results)


@app.route("/trainfinal")
def trainfinal():

    session = Session(engine)
    results = []
    result = session.execute(Trainfinal.__table__.select())
    for row in result:
        results.append(dict(row))
    session.close()

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)