from datetime import datetime

from app import db

class Album_nominates(db.Model):
    __tablename__ = 'album_nominates'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nominate_name = db.Column(db.String(100), index=True)
    nominate_url=db.Column(db.String(100))
    explored=db.Column(db.BOOLEAN,default=False)

album_nominate_relation = db.Table('album_nominate_relation',
    db.Column('album_nominate_id', db.Integer, db.ForeignKey('album_nominates.id')),
    db.Column('album_id', db.Integer, db.ForeignKey('albums.id'))
)

class Album_reviews(db.Model):
    __tablename__ = 'album_reviews'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    review = db.Column(db.String(1000), index=True)
    rating= db.Column(db.DECIMAL(5,2), default=0)

album_review_relation = db.Table('album_review_relation',
    db.Column('album_review_id', db.Integer, db.ForeignKey('album_reviews.id')),
    db.Column('album_id', db.Integer, db.ForeignKey('albums.id'))
)

class Albums(db.Model):
    __tablename__ = 'albums'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    album_name = db.Column(db.String(100), index=True)
    publish_year = db.Column(db.DateTime, default=datetime(1800, 1, 1))
    album_wiki=db.Column(db.String(3000),default="blank",index=True)
    reviews=db.relationship('Album_reviews', secondary=album_review_relation,
                               backref=db.backref('review_albums', lazy='dynamic'))
    album_rating=db.Column(db.DECIMAL(5,2), default=0)
    rate_number=db.Column(db.Integer, default=0)
    music = db.relationship('Music', backref='the_album', lazy='dynamic')
    nominate = db.relationship('Album_nominates', secondary=album_nominate_relation,
                               backref=db.backref('nominate_albums', lazy='dynamic'))
class Tribute(db.Model):
    __tablename__ = 'tribute'
    tributed_artist_id = db.Column(db.Integer, db.ForeignKey('artist.id'), primary_key=True)
    tribute_to_artist_id = db.Column(db.Integer, db.ForeignKey('artist.id'), primary_key=True)

class Teach(db.Model):
    __tablename__ = 'teach'
    teacher_id = db.Column(db.Integer, db.ForeignKey('artist.id'), primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('artist.id'), primary_key=True)

class Tours(db.Model):
    __tablename__ = 'tours'
    id=db.Column(db.Integer, primary_key=True,autoincrement = True)
    tour_name=db.Column(db.String(100), index=True)

class Nominates(db.Model):
    __tablename__ = 'nominates'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nominate_name = db.Column(db.String(100), index=True)

class Bands(db.Model):
    __tablename__ = 'bands'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    band_name = db.Column(db.String(100), index=True)

tour_relation = db.Table('tour_relation',
    db.Column('tour_id', db.Integer, db.ForeignKey('tours.id')),
    db.Column('artist_id', db.Integer, db.ForeignKey('artist.id'))
)

nominate_relation = db.Table('nominate_relation',
    db.Column('nominate_id', db.Integer, db.ForeignKey('nominates.id')),
    db.Column('artist_id', db.Integer, db.ForeignKey('artist.id'))
)

band_relation = db.Table('band_relation',
    db.Column('band_id', db.Integer, db.ForeignKey('bands.id')),
    db.Column('artist_id', db.Integer, db.ForeignKey('artist.id'))
)

album_relation = db.Table('album_relation',
    db.Column('album_id', db.Integer, db.ForeignKey('albums.id')),
    db.Column('artist_id', db.Integer, db.ForeignKey('artist.id'))
)
class Artist(db.Model):
    __tablename__ = 'artist'
    id = db.Column(db.Integer, primary_key=True,autoincrement = True)
    artist_name= db.Column(db.String(50), index=True)
    wikipedia=db.Column(db.String(3000),default="blank",index=True)
    tags = db.Column(db.String(1000),default="blank",index=True)
    gender = db.Column(db.String(10),default="blank",index=True)
    Born =db.Column(db.DateTime, default=datetime(1800,1,1))
    tribute_to = db.relationship('Tribute',
                                 foreign_keys=[Tribute.tribute_to_artist_id],
                                 backref=db.backref('tribute_to_artist', lazy='joined'),
                                 lazy='dynamic',
                                 cascade='all, delete-orphan')
    tributed = db.relationship('Tribute',
                               foreign_keys=[Tribute.tributed_artist_id],
                               backref=db.backref('tributed_artist', lazy='joined'),
                               lazy='dynamic',
                               cascade='all, delete-orphan')
    teachers = db.relationship('Teach',
                                 foreign_keys=[Teach.student_id],
                                 backref=db.backref('student', lazy='joined'),
                                 lazy='dynamic',
                                 cascade='all, delete-orphan')
    students = db.relationship('Teach',
                               foreign_keys=[Teach.teacher_id],
                               backref=db.backref('teacher', lazy='joined'),
                               lazy='dynamic',
                               cascade='all, delete-orphan')

    tour=db.relationship('Tours', secondary=tour_relation,
        backref=db.backref('tour_artists', lazy='dynamic'))
    nominate = db.relationship('Nominates', secondary=nominate_relation,
                           backref=db.backref('nominate_artists', lazy='dynamic'))
    band = db.relationship('Bands', secondary=band_relation,
                               backref=db.backref('band_artists', lazy='dynamic'))
    album= db.relationship('Albums', secondary=album_relation,
                               backref=db.backref('album_artists', lazy='dynamic'))
class Guitarists(db.Model):
    __tablename__ = 'guitarists'
    id=db.Column(db.Integer, primary_key=True,autoincrement = True)
    guitarist_name=db.Column(db.String(100), index=True)
    wikipedia = db.Column(db.String(3000), default="blank", index=True)
    gender = db.Column(db.String(10), default="blank", index=True)
    Born = db.Column(db.DateTime, default=datetime(1800, 1, 1))

class Writers(db.Model):
    __tablename__ = 'writers'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    writer_name = db.Column(db.String(100), index=True)
    wikipedia = db.Column(db.String(3000), default="blank", index=True)
    gender = db.Column(db.String(10), default="blank", index=True)
    Born = db.Column(db.DateTime, default=datetime(1800, 1, 1))

class Drummers(db.Model):
    __tablename__ = 'drummers'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    drummer_name = db.Column(db.String(100), index=True)
    wikipedia = db.Column(db.String(3000), default="blank", index=True)
    gender = db.Column(db.String(10), default="blank", index=True)
    Born = db.Column(db.DateTime, default=datetime(1800, 1, 1))

class Pianists(db.Model):
    __tablename__ = 'pianists'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pianist_name = db.Column(db.String(100), index=True)
    wikipedia = db.Column(db.String(3000), default="blank", index=True)
    gender = db.Column(db.String(10), default="blank", index=True)
    Born = db.Column(db.DateTime, default=datetime(1800, 1, 1))

class Mixers(db.Model):
    __tablename__ = 'mixers'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mixer_name = db.Column(db.String(100), index=True)
    wikipedia = db.Column(db.String(3000), default="blank", index=True)
    gender = db.Column(db.String(10), default="blank", index=True)
    Born = db.Column(db.DateTime, default=datetime(1800, 1, 1))

class Electric_guitarists(db.Model):
    __tablename__ = 'electric_guitarists'
    id=db.Column(db.Integer, primary_key=True,autoincrement = True)
    electric_guitarist_name=db.Column(db.String(100), index=True)
    wikipedia = db.Column(db.String(3000), default="blank", index=True)
    gender = db.Column(db.String(10), default="blank", index=True)
    Born = db.Column(db.DateTime, default=datetime(1800, 1, 1))

electric_guitarist_relation = db.Table('electric_guitarist_relation',
    db.Column('electric_guitarist_id', db.Integer, db.ForeignKey('electric_guitarists.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)

mixer_relation = db.Table('mixer_relation',
    db.Column('mixer_id', db.Integer, db.ForeignKey('mixers.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)

pianist_relation = db.Table('pianist_relation',
    db.Column('pianist_id', db.Integer, db.ForeignKey('pianists.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)

drummer_relation = db.Table('drummer_relation',
    db.Column('drummer_id', db.Integer, db.ForeignKey('drummers.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)

writer_relation = db.Table('writer_relation',
    db.Column('writer_id', db.Integer, db.ForeignKey('writers.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)
guitarist_relation = db.Table('guitarist_relation',
    db.Column('guitarist_id', db.Integer, db.ForeignKey('guitarists.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)
class Music_reviews(db.Model):
    __tablename__ = 'music_reviews'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    review = db.Column(db.String(1000), index=True)
    rating = db.Column(db.DECIMAL(5, 2), default=0)

music_review_relation = db.Table('music_review_relation',
    db.Column('music_review_id', db.Integer, db.ForeignKey('music_reviews.id')),
    db.Column('music_id', db.Integer, db.ForeignKey('music.id'))
)
class Music_tag(db.Model):
    __tablename__ = 'music_tag'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    tag_name=db.Column(db.String(40), index=True)
    tag_s_musics = db.relationship("Music_and_tag_relation", back_populates="musics")

class Music_and_tag_relation(db.Model):
    __tablename__ = 'music_and_tag_relation'
    music_id = db.Column(db.Integer, db.ForeignKey('music.id'), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('music_tag.id'), primary_key=True)
    weight=db.Column(db.Integer)
    musics = db.relationship("Music_tag", back_populates="tag_s_musics")
    tags = db.relationship("Music", back_populates="music_s_tags")

class Music(db.Model):
    __tablename__ = 'music'
    id = db.Column(db.Integer, primary_key=True,autoincrement = True)
    music_name= db.Column(db.String(100), index=True)
    music_story=db.Column(db.String(3000),default="blank",index=True)
    lyric=db.Column(db.String(3000),default="blank",index=True)
    album_id = db.Column(db.Integer, db.ForeignKey('albums.id'))
    mixer= db.relationship('Mixers', secondary=mixer_relation,
                           backref=db.backref('mixer_musics', lazy='dynamic'))
    writer = db.relationship('Writers', secondary=writer_relation,
                           backref=db.backref('writer_musics', lazy='dynamic'))
    drummer = db.relationship('Drummers', secondary=drummer_relation,
                               backref=db.backref('drummer_musics', lazy='dynamic'))
    pianist = db.relationship('Pianists', secondary=pianist_relation,
                           backref=db.backref('pianist_musics', lazy='dynamic'))
    guitarist = db.relationship('Guitarists', secondary=guitarist_relation,
                            backref=db.backref('guitarist_musics', lazy='dynamic'))

    electric_guitarist = db.relationship('Electric_guitarists', secondary=electric_guitarist_relation,
                            backref=db.backref('electric_guitarist_musics', lazy='dynamic'))

    include_mixer=db.Column(db.BOOLEAN,default=False)
    include_drummer = db.Column(db.BOOLEAN, default=False)
    include_pianist = db.Column(db.BOOLEAN, default=False)
    include_guitarist = db.Column(db.BOOLEAN, default=False)
    include_electric_guitarist = db.Column(db.BOOLEAN, default=False)

    audio_file_name=db.Column(db.String(100), default="blank", index=True)
    yt_link=db.Column(db.String(100), default="blank", index=True)
    reviews=db.relationship('Music_reviews', secondary=music_review_relation,
                               backref=db.backref('review_music', lazy='dynamic'))
    music_s_tags=db.relationship("Music_and_tag_relation", back_populates="tags")

class Search_list(db.Model):
    __tablename__ = 'search_list'
    nominate_url = db.Column(db.String(100),primary_key=True)
    explored = db.Column(db.BOOLEAN, default=False)