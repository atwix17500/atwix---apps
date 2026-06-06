import numpy as np
import joblib
import streamlit as st

from auth.auth_manager import AuthManager

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load ML model (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/crop_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, label_encoder


# ── Global styles ────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(27, 94, 32, 0.3);
        }
        .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
        .main-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; }

        .auth-card {
            background: #ffffff;
            border: 1px solid #e8f5e9;
            border-radius: 20px;
            padding: 0 2.25rem 2.25rem;
            box-shadow: 0 8px 32px rgba(27, 94, 32, 0.08);
            max-width: 520px;
            margin: 0 auto;
            overflow: hidden;
        }

        .auth-welcome-banner {
            background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #43a047 100%);
            margin: 0 -2.25rem 1.25rem;
            padding: 0.85rem 1rem;
            text-align: center;
            position: relative;
        }

        .auth-welcome-text {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.45rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            line-height: 1.2;
            letter-spacing: 0.03em;
            text-shadow: 0 1px 8px rgba(0, 0, 0, 0.2);
        }

        .auth-welcome-tagline {
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.88);
            margin: 0.25rem 0 0;
            font-weight: 400;
        }

        .auth-toggle {
            background: #eef6ee;
            border-radius: 14px;
            padding: 6px;
            margin-bottom: 1.75rem;
        }

        .auth-toggle div[data-testid="column"] .stButton > button {
            height: 46px;
            border-radius: 10px !important;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
            transition: all 0.2s ease;
            border: none !important;
        }

        .auth-toggle div[data-testid="column"] .stButton > button[kind="secondary"] {
            background: transparent !important;
            color: #558b2f !important;
            box-shadow: none !important;
        }

        .auth-toggle div[data-testid="column"] .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.55) !important;
            color: #2e7d32 !important;
        }

        .auth-toggle div[data-testid="column"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2e7d32, #43a047) !important;
            color: white !important;
            box-shadow: 0 4px 14px rgba(46, 125, 50, 0.35) !important;
        }

        .auth-toggle div[data-testid="column"] .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
        }

        .auth-form-heading {
            font-size: 1.05rem;
            font-weight: 600;
            color: #2e7d32;
            margin: 0 0 0.15rem;
        }

        .auth-form-caption {
            color: #777;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stForm"] .stFormSubmitButton > button {
            background: linear-gradient(135deg, #2e7d32, #43a047) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            height: 48px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            margin-top: 0.75rem;
            box-shadow: 0 4px 14px rgba(46, 125, 50, 0.3);
        }

        div[data-testid="stForm"] .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
            box-shadow: 0 6px 18px rgba(46, 125, 50, 0.4);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            border-radius: 10px !important;
            border-color: #c8e6c9 !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus {
            border-color: #2e7d32 !important;
            box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.15) !important;
        }

        .demo-box {
            background: #f1f8e9;
            border: 1px solid #dcedc8;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            font-size: 0.88rem;
            color: #33691e;
            text-align: center;
            margin-top: 1rem;
        }

        .password-rules {
            background: #fafafa;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            font-size: 0.84rem;
            color: #555;
            border: 1px solid #eee;
            margin-top: 1rem;
        }

        .password-rules strong {
            color: #2e7d32;
        }

        .rule-list { font-size: 0.85rem; color: #555; }
        .rule-ok   { color: #2e7d32; }
        .rule-fail { color: #c62828; }

        .result-box {
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            border-left: 5px solid #2e7d32;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-top: 1rem;
        }
        .result-box h2 { color: #1b5e20; margin: 0; font-size: 1.6rem; }

        .metric-card {
            background: #f9fbe7;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
            border: 1px solid #dce775;
        }
        .metric-card .label { font-size: 0.75rem; color: #558b2f; text-transform: uppercase; letter-spacing: 0.05em; }
        .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #33691e; }

        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2e7d32, #43a047);
            border: none;
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1b5e20, #2e7d32);
        }

        #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Session helpers ──────────────────────────────────────────────────────────
auth = AuthManager()


def init_session():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"


def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()


def require_auth():
    init_session()
    if not st.session_state.authenticated or not auth.is_session_valid(st.session_state.user):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.warning("Your session has expired. Please log in again.")
        st.stop()
    if "user_id" not in st.session_state.user:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.warning("Please log in again to continue.")
        st.stop()


# ── Countries list for registration ──────────────────────────────────────────
COUNTRIES = [
    "Select country",
    "Uganda",
    "Kenya",
    "Tanzania",
    "Rwanda",
    "Ethiopia",
    "Nigeria",
    "Ghana",
    "South Africa",
    "India",
    "Pakistan",
    "Bangladesh",
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "Other",
]


# ── Page: Login & Register ───────────────────────────────────────────────────
def auth_page():
    init_session()
    inject_styles()
    st.markdown(
        """
        <div class="main-header" style="text-align:center;">
            <h1>🌾 Crop Recommendation System</h1>
            <p>Sign in or create an account to get crop suggestions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        welcome_tagline = (
            "Glad to see you again — sign in to continue"
            if st.session_state.auth_mode == "login"
            else "Join us and grow smarter every season"
        )

        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="auth-welcome-banner">
                <h2 class="auth-welcome-text">Welcome</h2>
                <p class="auth-welcome-tagline">{welcome_tagline}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="auth-toggle">', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2, gap="small")
        with btn_col1:
            if st.button(
                "🔐  Sign In",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "login" else "secondary",
                key="toggle_login",
            ):
                st.session_state.auth_mode = "login"
                st.rerun()
        with btn_col2:
            if st.button(
                "📝  Register",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "register" else "secondary",
                key="toggle_register",
            ):
                st.session_state.auth_mode = "register"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.markdown('<p class="auth-form-heading">Sign in to your account</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="auth-form-caption">Enter your credentials below</p>',
                unsafe_allow_html=True,
            )

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username", key="login_user")
                password = st.text_input(
                    "Password", type="password", placeholder="Enter your password", key="login_pass"
                )
                submitted = st.form_submit_button("Sign In →", use_container_width=True)

                if submitted:
                    success, message, user_data = auth.login(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = user_data
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

            st.markdown(
                '<div class="demo-box">Demo account: <strong>admin</strong> · password <strong>Admin@123</strong></div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown('<p class="auth-form-heading">Create your account</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="auth-form-caption">Fill in all fields to get started</p>',
                unsafe_allow_html=True,
            )

            with st.form("register_form"):
                username = st.text_input("Username", placeholder="e.g. farmer_john")
                email = st.text_input("Email", placeholder="you@example.com")

                loc1, loc2 = st.columns(2)
                with loc1:
                    country = st.selectbox("Country", COUNTRIES)
                with loc2:
                    district = st.text_input(
                        "District / Region",
                        placeholder="e.g. Kampala, Punjab",
                    )

                password = st.text_input("Password", type="password", placeholder="Create a strong password")
                confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                submitted = st.form_submit_button("Create Account →", use_container_width=True)

                if submitted:
                    success, message = auth.register(
                        username, email, password, confirm, country, district
                    )
                    if success:
                        st.success(message + " Use the **Sign In** button above to log in.")
                    else:
                        st.error(message)

            st.markdown(
                """
                <div class="password-rules">
                    <strong>Password requirements</strong><br>
                    • At least 8 characters &nbsp;• One uppercase letter<br>
                    • One lowercase letter &nbsp;• One number &nbsp;• One special character
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Dashboard ──────────────────────────────────────────────────────────
def dashboard_page():
    require_auth()
    inject_styles()
    user = st.session_state.user

    st.markdown(
        f"""
        <div class="main-header">
            <h1>🌾 Crop Recommendation Dashboard</h1>
            <p>Welcome, <strong>{user['username']}</strong> · Enter soil & climate data to get crop suggestions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, label_encoder = load_model()

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Soil & Environment Inputs")
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
                P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0)
                K = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0)
            with c2:
                temperature = st.number_input("Temperature (°C)", value=20.8, step=0.1)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
                ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=202.9, step=0.1)
            predict_btn = st.form_submit_button("🔍 Get Recommendation", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Results")
        if predict_btn:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            crop_name = label_encoder.inverse_transform(prediction)[0]

            auth.save_prediction(
                user["user_id"],
                N, P, K, temperature, humidity, ph, rainfall, crop_name,
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <p style="margin:0;color:#558b2f;font-size:0.9rem;">Recommended Crop</p>
                    <h2>🌱 {crop_name}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            for col, label, val in [
                (m1, "N", f"{N:.0f}"),
                (m2, "P", f"{P:.0f}"),
                (m3, "K", f"{K:.0f}"),
                (m4, "pH", f"{ph:.1f}"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{val}</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Fill in the soil and climate values, then click **Get Recommendation**.")


# ── Page: Profile ────────────────────────────────────────────────────────────
def profile_page():
    require_auth()
    inject_styles()
    user = st.session_state.user
    remaining = auth.get_session_remaining(user)

    st.markdown(
        """
        <div class="main-header">
            <h1>👤 My Profile</h1>
            <p>Manage your account and session</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    profile = auth.get_user_profile(user["user_id"])

    with col1:
        st.subheader("Account Details")
        st.markdown(f"**Username:** {user['username']}")
        st.markdown(f"**Email:** {user['email']}")
        st.markdown(f"**Country:** {user.get('country', 'Not set')}")
        st.markdown(f"**District / Region:** {user.get('district', 'Not set')}")
        st.markdown(f"**Role:** {user['role'].capitalize()}")
        if profile and profile.get("created_at"):
            st.markdown(f"**Member since:** {profile['created_at'][:19].replace('T', ' ')}")
        st.markdown(f"**Logged in since:** {user['login_time'][:19].replace('T', ' ')}")
        st.markdown(f"**Session expires in:** {remaining} minutes")

    with col2:
        st.subheader("Session")
        if remaining < 5:
            st.warning("Your session is about to expire. Save your work and re-login if needed.")
        else:
            st.success(f"Session active — {remaining} minutes remaining.")

        if st.button("Sign Out", type="primary", use_container_width=True):
            logout()

    st.divider()
    st.subheader("Recent Predictions")
    history = auth.get_user_predictions(user["user_id"], limit=10)
    if history:
        for row in history:
            date = row["created_at"][:19].replace("T", " ")
            st.markdown(
                f"**{row['recommended_crop']}** · N={row['nitrogen']:.0f}, "
                f"P={row['phosphorus']:.0f}, K={row['potassium']:.0f} · {date}"
            )
    else:
        st.caption("No predictions yet. Run one from the Dashboard.")


# ── Navigation ───────────────────────────────────────────────────────────────
init_session()

if not st.session_state.authenticated or not auth.is_session_valid(st.session_state.user):
    st.session_state.authenticated = False
    st.session_state.user = None
    auth_page()
else:
    with st.sidebar:
        st.markdown(f"**Signed in as** {st.session_state.user['username']}")
        if st.button("Sign Out", use_container_width=True):
            logout()

    pages = st.navigation(
        [
            st.Page(dashboard_page, title="Dashboard", icon="🌾", default=True),
            st.Page(profile_page, title="Profile", icon="👤"),
        ]
    )
    pages.run()
