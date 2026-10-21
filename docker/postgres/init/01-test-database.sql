-- Runs once, when the Postgres volume is first created. Integration tests use this database
-- (TEST_DATABASE_URL) and wipe it freely; the application database is never touched by tests.
CREATE DATABASE retrieval_labs_test OWNER retrieval_labs;
