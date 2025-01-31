from sqlalchemy import Column, Integer, String, Date, Text, create_engine, desc
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import date

Base = declarative_base()

class Campaign(Base):
    __tablename__ = "campaigns"

    campaign_id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_type = Column(String(50), nullable=False)
    campaign_date = Column(Date, default=date.today, nullable=False)
    campaign_name = Column(String(100), nullable=False)
    segment = Column(String(255), nullable=False)
    campaign_text = Column(Text, nullable=False)
    campaign_jiratask = Column(String(50), nullable=True)

    def __init__(self, campaign_type, campaign_name, segment, campaign_text, campaign_jiratask=None):
        self.campaign_type = campaign_type
        self.campaign_date = date.today()
        self.campaign_name = campaign_name
        self.segment = segment
        self.campaign_text = campaign_text
        self.campaign_jiratask = campaign_jiratask

    def get_latest_campaign(self):
        # Database connection
        DATABASE_URL = "sqlite:///campaigns.db"
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Fetch the latest campaign entry
        return session.query(Campaign).order_by(desc(Campaign.campaign_id)).first()