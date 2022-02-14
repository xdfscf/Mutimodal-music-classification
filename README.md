# flask_project_new
The aim of the project is to build a recommendation system based on useful information of music.
The system may need a similarity analysis algorithm that consider both genre classification and background stories/lyrics similarity.

The first step of the project is to retrieve information from following websites -

Musicbrainz: basic information of artists and the relationship between artists. The website also provides the information of artists' albums and single, which is useful as the foundation of aggregation. The data could be processed by cluster methods or nlp models.

Rateyourmusic: The website provides the ratings and reviews of single and albums. The data could be processed by nlp emotion models

Songfacts: The interesting facts of single are posted in the website. The data could be processed by nlp emotion or similarity models.(Hard to aviod the hcaptcha system)

genius: provides lyrics, also could be processed by nlp emotion or similarity models.

A database was already built to save the retrieved information.

The second step is to introduce deep learning methods to analysis retrieved data

The last step is to build a user interface that can show the recommendation information and probably the reason of the recommondation.(data visualization)
The flask was chosen to construct the user interface.
