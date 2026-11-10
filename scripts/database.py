import os

if os.getenv("PGHOST"):
    from database_postgres import create_table, get_history, save_prediction
else:
    from database_sqlite import create_table, get_history, save_prediction
