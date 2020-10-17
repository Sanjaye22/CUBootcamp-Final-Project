import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, cast, Integer
from flask import Flask, render_template, redirect, jsonify, request
import pandas as pd

# database setup
# Configure settings for RDS
rds_connection_string = "root:RamNivas08@predictyoursalary.cruwlreaual5.us-east-2.rds.amazonaws.com:5432/predictyoursalary"
engine = create_engine(f'postgresql://{rds_connection_string}')

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)