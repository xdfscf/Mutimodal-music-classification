"""empty message

Revision ID: 10bec1d64a76
Revises: c69cc1cc9757
Create Date: 2022-02-05 12:21:54.276890

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '10bec1d64a76'
down_revision = 'c69cc1cc9757'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('album_nominates',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('nominate_name', sa.String(length=100), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_album_nominates_nominate_name'), 'album_nominates', ['nominate_name'], unique=False)
    op.create_table('album_nominate_relation',
    sa.Column('album_nominate_id', sa.Integer(), nullable=True),
    sa.Column('album_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['album_id'], ['albums.id'], ),
    sa.ForeignKeyConstraint(['album_nominate_id'], ['album_nominates.id'], )
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('album_nominate_relation')
    op.drop_index(op.f('ix_album_nominates_nominate_name'), table_name='album_nominates')
    op.drop_table('album_nominates')
    # ### end Alembic commands ###
