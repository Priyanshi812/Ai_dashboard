import sqlite3

# Connect (creates the file if not exists)
conn = sqlite3.connect("factory.db")
cur = conn.cursor()

# Create tables
cur.execute("""
CREATE TABLE IF NOT EXISTS sku (
    sku_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    size TEXT,
    finish TEXT,
    unit TEXT DEFAULT 'pcs',
    price REAL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS raw_material (
    material_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    uom TEXT DEFAULT 'kg',
    lead_time_days INTEGER DEFAULT 7,
    cost REAL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS inventory_txn (
    txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type TEXT CHECK(item_type IN ('sku','material')) NOT NULL,
    item_id INTEGER NOT NULL,
    qty REAL NOT NULL,
    reason TEXT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Insert sample data only if empty
cur.execute("SELECT COUNT(*) FROM sku")
if cur.fetchone()[0] == 0:
    cur.executescript("""
    INSERT INTO sku (name, size, finish, unit, price) VALUES
    ('Tile A', '12x12', 'Glossy', 'pcs', 30),
    ('Tile B', '24x24', 'Matte', 'pcs', 50);

    INSERT INTO raw_material (name, uom, lead_time_days, cost) VALUES
    ('Clay', 'kg', 5, 10),
    ('Glaze', 'kg', 7, 25);

    INSERT INTO inventory_txn (item_type, item_id, qty, reason) VALUES
    ('sku', 1, 100, 'Initial stock'),
    ('sku', 2, 50, 'Initial stock'),
    ('material', 1, 500, 'Purchase'),
    ('material', 2, 200, 'Purchase'),
    ('sku', 1, -20, 'Sales Order #101'),
    ('material', 1, -50, 'Used in production');
    """)
    print("Inserted sample data ✅")

conn.commit()
conn.close()
print("Database setup complete ✅")
