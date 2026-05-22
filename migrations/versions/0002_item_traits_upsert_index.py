"""item traits upsert index

Revision ID: 0002_item_traits_upsert_index
Revises: 0001_data_spine_pgvector
Create Date: 2026-05-21
"""

from alembic import op


revision = "0002_item_traits_upsert_index"
down_revision = "0001_data_spine_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_item_traits_canonical;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_item_traits_canonical_unique
          ON item_traits (canonical_item_id, prompt_version);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_item_traits_canonical_unique;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_item_traits_canonical
          ON item_traits (canonical_item_id, prompt_version)
          WHERE canonical_item_id IS NOT NULL;
        """
    )
