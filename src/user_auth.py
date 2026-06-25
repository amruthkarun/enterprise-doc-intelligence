USER_PERMISSIONS = {
    "alice@email.com": ["Allianz_Earnings_Full_Transcript"],
    "amruthkarun@gmail.com": ["Allianz_Earnings_Full_Transcript"],
    "bob@email.com": ["Apple_Earnings_Full_Transcript", "Corporate_Earnings_Transcripts_2026"],
    "charlie@email.com": ["JIO_Earnings_Full_Transcript"]
}


def get_user_access(email):
    return USER_PERMISSIONS.get(email, [])
