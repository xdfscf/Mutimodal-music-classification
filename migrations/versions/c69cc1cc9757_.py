"""empty message

Revision ID: c69cc1cc9757
Revises: 2533b81e6925
Create Date: 2022-02-03 12:34:00.145932

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c69cc1cc9757'
down_revision = '2533b81e6925'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('guitarists', sa.Column('wikipedia', sa.String(length=3000), nullable=True))
    op.add_column('guitarists', sa.Column('gender', sa.String(length=10), nullable=True))
    op.add_column('guitarists', sa.Column('Born', sa.DateTime(), nullable=True))
    op.create_index(op.f('ix_guitarists_gender'), 'guitarists', ['gender'], unique=False)
    op.create_index(op.f('ix_guitarists_wikipedia'), 'guitarists', ['wikipedia'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_guitarists_wikipedia'), table_name='guitarists')
    op.drop_index(op.f('ix_guitarists_gender'), table_name='guitarists')
    op.drop_column('guitarists', 'Born')
    op.drop_column('guitarists', 'gender')
    op.drop_column('guitarists', 'wikipedia')
    # ### end Alembic commands ###
