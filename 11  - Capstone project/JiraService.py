import requests
from requests.auth import HTTPBasicAuth

def create_jira_task(Campaign):
    # Replace with your Jira credentials and endpoint
    JIRA_URL = "https://jira.uztelecom.uz/rest/api/2/issue"
    USERNAME = ""
    PASSWORD = ""

    PROJECT_KEY = "MRKT"
    SUMMARY = "SMS рассылки на " + Campaign.campaign_date
    TEXT = Campaign.campaign_text

    ASSIGNEE = "r.latipov"
    ASSIGNEE_NAME = ""
    DESCRIPTION = ("SMS рассылка на всю базу абонентов - \n\n" + Campaign.campaign_name +
                   "Альфа номер *Strategy*\n\n"
                   "*EN:*\n\n " + TEXT + "\n\n\n")


    CO_AUTHORS = ("a.bobokhonov, a.ishkobilov, b.omonov, d.dkadirova, d.karimberdiev, "
                  "d.khaydarshikov, d.xakimov, d.yrasulova, dj.adilov, f.fabdullaev, "
                  "j.islamov, k.kattaev, kh.akhmedjanov, m.miryunusova, m.rruzieva, "
                  "m.shishkin, n.abdurakhmanova, n.abduvahobov, n.safarov, r.latipov, "
                  "s.ismailova, s.turapov, sa.ibragimova, sh.shamansurov, "
                  "sh.shyusupov, shm.umarov")

    # Define the task payload
    payload = {
        "fields": {
            "project": {
                "key": PROJECT_KEY
            },
            "summary": SUMMARY,
            "description": DESCRIPTION,
            "issuetype": {
                "name": "Задача"
            },
            "assignee": {
                "key": ASSIGNEE,
                "name": ASSIGNEE
            },
            "duedate": "2025-02-01",
            "customfield_10032": [
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=n.abduvahobov', 'name': 'n.abduvahobov',
                 'key': 'n.abduvahobov', 'emailAddress': 'n.abduvahobov@utc.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=n.abduvahobov&avatarId=16524',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=n.abduvahobov&avatarId=16524',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=n.abduvahobov&avatarId=16524',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=n.abduvahobov&avatarId=16524'},
                 'displayName': 'Абдувахобов Нодирбек [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=f.fabdullaev', 'name': 'f.fabdullaev',
                 'key': 'f.fabdullaev', 'emailAddress': 'f.abdullayev@noctelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/1e4c4a6da2bb9a07b8527538deea8561?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/1e4c4a6da2bb9a07b8527538deea8561?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/1e4c4a6da2bb9a07b8527538deea8561?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/1e4c4a6da2bb9a07b8527538deea8561?d=mm&s=32'},
                 'displayName': 'Абдуллаев Фаррух [СТФ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=n.abdurakhmanova', 'name': 'n.abdurakhmanova',
                 'key': 'n.abdurakhmanova', 'emailAddress': 'n.abdurakhmanova@utc.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/57fc381f7c1f576995ca5546e5715186?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/57fc381f7c1f576995ca5546e5715186?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/57fc381f7c1f576995ca5546e5715186?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/57fc381f7c1f576995ca5546e5715186?d=mm&s=32'},
                 'displayName': 'Абдурахманова Наргиза [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=dj.adilov', 'name': 'dj.adilov',
                 'key': 'dj.adilov', 'emailAddress': 'dj.adilov@uztelecom.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=dj.adilov&avatarId=11343',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=dj.adilov&avatarId=11343',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=dj.adilov&avatarId=11343',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=dj.adilov&avatarId=11343'},
                 'displayName': 'Адилов Джахангир\xa0[АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=kh.akhmedjanov', 'name': 'kh.akhmedjanov',
                 'key': 'kh.akhmedjanov', 'emailAddress': 'x.axmedjanov@noctelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/04a6ea994069eae28113b49e3e4fb6d4?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/04a6ea994069eae28113b49e3e4fb6d4?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/04a6ea994069eae28113b49e3e4fb6d4?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/04a6ea994069eae28113b49e3e4fb6d4?d=mm&s=32'},
                 'displayName': 'Ахмеджанов Хуршид [СТФ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=a.bobokhonov', 'name': 'a.bobokhonov',
                 'key': 'a.bobokhonov', 'emailAddress': 'a.bobokhonov@utc.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=a.bobokhonov&avatarId=15306',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=a.bobokhonov&avatarId=15306',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=a.bobokhonov&avatarId=15306',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=a.bobokhonov&avatarId=15306'},
                 'displayName': 'Бобохонов Аброр [КЦ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=sa.ibragimova', 'name': 'sa.ibragimova',
                 'key': 's.ibragimov', 'emailAddress': 'sa.ibragimova@utc.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=s.ibragimov&avatarId=12103',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=s.ibragimov&avatarId=12103',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=s.ibragimov&avatarId=12103',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=s.ibragimov&avatarId=12103'},
                 'displayName': 'Ибрагимова Саджида [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=j.islamov', 'name': 'j.islamov',
                 'key': 'j.islamov', 'emailAddress': 'j.islamov@uztelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/5e2884d4fe0d4d1516e2eaf299f53713?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/5e2884d4fe0d4d1516e2eaf299f53713?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/5e2884d4fe0d4d1516e2eaf299f53713?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/5e2884d4fe0d4d1516e2eaf299f53713?d=mm&s=32'},
                 'displayName': 'Исламов Жавлон [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=s.ismailova', 'name': 's.ismailova',
                 'key': 'ID15804', 'emailAddress': 's.ismailova@uztelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/51086bec361e3e910cdc8348643f0e5f?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/51086bec361e3e910cdc8348643f0e5f?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/51086bec361e3e910cdc8348643f0e5f?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/51086bec361e3e910cdc8348643f0e5f?d=mm&s=32'},
                 'displayName': 'Исмаилова Сабина [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=a.ishkobilov', 'name': 'a.ishkobilov',
                 'key': 'a.ishkobilov', 'emailAddress': 'b2b7@markaziy.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/8aaf9ab326193484a51773e70b36bbb3?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/8aaf9ab326193484a51773e70b36bbb3?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/8aaf9ab326193484a51773e70b36bbb3?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/8aaf9ab326193484a51773e70b36bbb3?d=mm&s=32'},
                 'displayName': 'Ишкобилов Азиз [MARKAZ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=d.dkadirova', 'name': 'd.dkadirova',
                 'key': 'JIRAUSER21409', 'emailAddress': 'd.kadirova@utc.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?avatarId=14601',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&avatarId=14601',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&avatarId=14601',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&avatarId=14601'},
                 'displayName': 'Кадирова Дилдора [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=d.karimberdiev', 'name': 'd.karimberdiev',
                 'key': 'd.karimberdiev', 'emailAddress': 'd.karimberdiev@uztelecom.uz', 'avatarUrls': {
                    '48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=d.karimberdiev&avatarId=15258',
                    '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=d.karimberdiev&avatarId=15258',
                    '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=d.karimberdiev&avatarId=15258',
                    '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=d.karimberdiev&avatarId=15258'},
                 'displayName': 'Каримбердиев Даврон [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=k.kattaev', 'name': 'k.kattaev',
                 'key': 'k.kattaev', 'emailAddress': 'murojaat5@markaziy.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/0d37c7c9c4c919c3e6d0363057bb5c3d?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/0d37c7c9c4c919c3e6d0363057bb5c3d?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/0d37c7c9c4c919c3e6d0363057bb5c3d?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/0d37c7c9c4c919c3e6d0363057bb5c3d?d=mm&s=32'},
                 'displayName': 'Каттаев Каттабой [MARKAZ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=r.latipov', 'name': 'r.latipov',
                 'key': 'r.latipov', 'emailAddress': 'r.latipov@noctelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/4929319933967b887dfe49ade7f4cfd8?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/4929319933967b887dfe49ade7f4cfd8?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/4929319933967b887dfe49ade7f4cfd8?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/4929319933967b887dfe49ade7f4cfd8?d=mm&s=32'},
                 'displayName': 'Латипов Рустам [СТФ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=m.miryunusova', 'name': 'm.miryunusova',
                 'key': 'm.miryunusova', 'emailAddress': 'm.miryunusova@utc.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/8b8f8c6f6cc09e8023919a5abae967cc?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/8b8f8c6f6cc09e8023919a5abae967cc?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/8b8f8c6f6cc09e8023919a5abae967cc?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/8b8f8c6f6cc09e8023919a5abae967cc?d=mm&s=32'},
                 'displayName': 'Мирюнусова Мадина [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=b.omonov', 'name': 'b.omonov',
                 'key': 'JIRAUSER22931', 'emailAddress': 'b.omonov@noctelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/9dd3a02e5404c3d0fd89db4ca6160e14?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/9dd3a02e5404c3d0fd89db4ca6160e14?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/9dd3a02e5404c3d0fd89db4ca6160e14?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/9dd3a02e5404c3d0fd89db4ca6160e14?d=mm&s=32'},
                 'displayName': 'Омонов Бекзод\xa0[СТФ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=d.yrasulova', 'name': 'd.yrasulova',
                 'key': 'd.yrasulova', 'emailAddress': 'd.yrasulova@utc.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/e85dd250daceff71d254bb408715128d?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/e85dd250daceff71d254bb408715128d?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/e85dd250daceff71d254bb408715128d?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/e85dd250daceff71d254bb408715128d?d=mm&s=32'},
                 'displayName': 'Расулова Дильноза [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=m.rruzieva', 'name': 'm.rruzieva',
                 'key': 'm.rruzieva', 'emailAddress': 'm.rruzieva@utc.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=m.rruzieva&avatarId=13005',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=m.rruzieva&avatarId=13005',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=m.rruzieva&avatarId=13005',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=m.rruzieva&avatarId=13005'},
                 'displayName': 'Рузиева Малика [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=n.safarov', 'name': 'n.safarov',
                 'key': 'n.safarov', 'emailAddress': 'n.safarov@uztelecom.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=n.safarov&avatarId=14911',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=n.safarov&avatarId=14911',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=n.safarov&avatarId=14911',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=n.safarov&avatarId=14911'},
                 'displayName': 'Сафаров Нодир [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=s.turapov', 'name': 's.turapov',
                 'key': 's.turapov', 'emailAddress': 's.turapov@telecomsoft.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/63e5c9142acfec4c4bfd8321ef433aca?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/63e5c9142acfec4c4bfd8321ef433aca?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/63e5c9142acfec4c4bfd8321ef433aca?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/63e5c9142acfec4c4bfd8321ef433aca?d=mm&s=32'},
                 'displayName': 'Турапов Сарвар [TS]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=shm.umarov', 'name': 'shm.umarov',
                 'key': 'sh.mumarov', 'emailAddress': 'shm.umarov@utc.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=sh.mumarov&avatarId=20120',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=sh.mumarov&avatarId=20120',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=sh.mumarov&avatarId=20120',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=sh.mumarov&avatarId=20120'},
                 'displayName': 'Умаров Шохруххон [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=d.khaydarshikov', 'name': 'd.khaydarshikov',
                 'key': 'd.khaydarshikov', 'emailAddress': 'd.khaydarshikov@utc.uz', 'avatarUrls': {
                    '48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=d.khaydarshikov&avatarId=11376',
                    '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=d.khaydarshikov&avatarId=11376',
                    '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=d.khaydarshikov&avatarId=11376',
                    '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=d.khaydarshikov&avatarId=11376'},
                 'displayName': 'Хайдаршиков Диёр [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=d.xakimov', 'name': 'd.xakimov',
                 'key': 'd.xakimov', 'emailAddress': 'd.khakimov@infosystems.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/a6f187027e606751d909d5359246c1c5?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/a6f187027e606751d909d5359246c1c5?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/a6f187027e606751d909d5359246c1c5?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/a6f187027e606751d909d5359246c1c5?d=mm&s=32'},
                 'displayName': 'Хакимов Даврон [TS]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=sh.shamansurov', 'name': 'sh.shamansurov',
                 'key': 'sh.shamansurov', 'emailAddress': 'sh.shamansurov@uztelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/b68178f35cc1137fe87bdb622a0abe65?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/b68178f35cc1137fe87bdb622a0abe65?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/b68178f35cc1137fe87bdb622a0abe65?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/b68178f35cc1137fe87bdb622a0abe65?d=mm&s=32'},
                 'displayName': 'Шамансуров Шоолим [АПП]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=m.shishkin', 'name': 'm.shishkin',
                 'key': 'JIRAUSER21906', 'emailAddress': 'm.shishkin@noctelecom.uz',
                 'avatarUrls': {'48x48': 'https://www.gravatar.com/avatar/584feff8506a27f0d87688e3969fc749?d=mm&s=48',
                                '24x24': 'https://www.gravatar.com/avatar/584feff8506a27f0d87688e3969fc749?d=mm&s=24',
                                '16x16': 'https://www.gravatar.com/avatar/584feff8506a27f0d87688e3969fc749?d=mm&s=16',
                                '32x32': 'https://www.gravatar.com/avatar/584feff8506a27f0d87688e3969fc749?d=mm&s=32'},
                 'displayName': 'Шишкин Михаил [СТФ]', 'active': True, 'timeZone': 'Asia/Tashkent'},
                {'self': 'https://jira.uztelecom.uz/rest/api/2/user?username=sh.shyusupov', 'name': 'sh.shyusupov',
                 'key': 'sh.shyusupov', 'emailAddress': 'yusupov@uztelecom.uz',
                 'avatarUrls': {'48x48': 'https://jira.uztelecom.uz/secure/useravatar?ownerId=sh.shyusupov&avatarId=17752',
                                '24x24': 'https://jira.uztelecom.uz/secure/useravatar?size=small&ownerId=sh.shyusupov&avatarId=17752',
                                '16x16': 'https://jira.uztelecom.uz/secure/useravatar?size=xsmall&ownerId=sh.shyusupov&avatarId=17752',
                                '32x32': 'https://jira.uztelecom.uz/secure/useravatar?size=medium&ownerId=sh.shyusupov&avatarId=17752'},
                 'displayName': 'Юсупов Шерзод [КЦ]', 'active': True, 'timeZone': 'Asia/Tashkent'}]

        }
    }

    # "customfield_10032": {
    #     {"key": "a.bobokhonov"}, {"key": "a.ishkobilov"}, {"key": "b.omonov"}, {"key": "d.dkadirova"},
    #     {"key": "d.karimberdiev"},
    #     {"key": "d.khaydarshikov"}, {"key": "d.xakimov"}, {"key": "d.yrasulova"}, {"key": "dj.adilov"},
    #     {"key": "f.fabdullaev"},
    #     {"key": "j.islamov"}, {"key": "k.kattaev"}, {"key": "kh.akhmedjanov"}, {"key": "m.miryunusova"},
    #     {"key": "m.rruzieva"},
    #     {"key": "m.shishkin"}, {"key": "n.abdurakhmanova"}, {"key": "n.abduvahobov"}, {"key": "n.safarov"},
    #     {"key": "r.latipov"},
    #     {"key": "s.ismailova"}, {"key": "s.turapov"}, {"key": "sa.ibragimova"}, {"key": "sh.gibragimov"},
    #     {"key": "sh.shamansurov"},
    #     {"key": "sh.shyusupov"}, {"key": "shm.umarov"}
    # }

    # Set headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request to create the task
    try:
        response = requests.post(
            JIRA_URL,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            json=payload,
            headers=headers
        )

        # Check if the task was created successfully
        if response.status_code == 201:
            print("Task created successfully!")
            print("Task Key:", response.json().get("key"))
        else:
            print("Failed to create task.")
            print("Status Code:", response.status_code)
            print("Response:", response.text)

    except Exception as e:
        print("An error occurred:", e)

