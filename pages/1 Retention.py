import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import calendar
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 윈도우 폰트 지정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


# from data import load_all_data
# all_data = load_all_data()


# ---- [내 load_all_data 대체 함수 추가] ----
@st.cache_data
def load_all_data(base_path="data/"):
    # 1. CSV 로드
    users = pd.read_csv(base_path + "users.csv")
    products = pd.read_csv(base_path + "products.csv")
    orders = pd.read_csv(base_path + "orders.csv")
    order_items = pd.read_csv(base_path + "order_items.csv")
    events = pd.read_csv(base_path + "events_sample.csv")
    inventory_items = pd.read_csv(base_path + "inventory_items.csv")

    # 2. 날짜 변환
    date_cols = {
        "users": ["created_at"],
        "orders": ["created_at", "returned_at", "shipped_at", "delivered_at"],
        "order_items": ["created_at", "shipped_at", "delivered_at", "returned_at"],
        "events": ["created_at"],
        "inventory_items": ["created_at", "sold_at"]
    }
    dfs = {
        "users": users,
        "orders": orders,
        "order_items": order_items,
        "events": events,
        "inventory_items": inventory_items
    }
    for df_name, cols in date_cols.items():
        for col in cols:
            dfs[df_name][col] = pd.to_datetime(dfs[df_name][col], errors="coerce")

    # 3. 2023 데이터만 필터링
    users = users[(users['created_at'] >= "2023-01-01") & (users['created_at'] <= "2023-12-31")]
    orders = orders[(orders['created_at'] >= "2023-01-01") & (orders['created_at'] <= "2023-12-31")]
    order_items = order_items[(order_items['created_at'] >= "2023-01-01") & (order_items['created_at'] <= "2023-12-31")]
    events = events[(events['created_at'] >= "2023-01-01") & (events['created_at'] <= "2023-12-31")]
    inventory_items = inventory_items[(inventory_items['created_at'] >= "2023-01-01") & (inventory_items['created_at'] <= "2023-12-31")]

    return {
        "users": users,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "events": events,
        "inventory_items": inventory_items
    }



@st.cache_data
def create_purchase_distribution_chart(order_items_df):
    """사용자별 구매 횟수 분포를 계산하고 막대그래프를 생성합니다."""
    
    # 'Complete' 상태인 주문만 필터링
    completed_orders = order_items_df[order_items_df['status'] == 'Complete']
    
    if completed_orders.empty:
        st.warning("분석할 완료된 주문 데이터가 없습니다.")
        return None, None
        
    # 사용자별 구매 횟수 계산 (user_id별로 그룹화 후 order_id의 고유 개수 계산)
    user_purchase_counts = completed_orders.groupby('user_id')['order_id'].nunique()
    
    # 구매 횟수별 사용자 수 분포 계산
    purchase_dist = user_purchase_counts.value_counts().sort_index()

    # --- Matplotlib 차트 생성 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    purchase_dist.plot(kind='bar', ax=ax, color='skyblue')
    
    ax.set_title("사용자별 구매 횟수 분포", fontsize=18, fontweight='bold')
    ax.set_xlabel("사용자당 총 구매 횟수", fontsize=12)
    ax.set_ylabel("사용자 수", fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    return fig, purchase_dist

@st.cache_data
def create_retention_heatmap(order_items_df, show_annotations=True):
    """
    주문 데이터를 기반으로 코호트 리텐션 분석을 수행하고 히트맵을 생성합니다.
    """
    # 'Complete' 상태인 주문만 필터링
    valid_orders = order_items_df[order_items_df['status'] == 'Complete'].copy()

    if valid_orders.empty:
        return None, None

    # --- 코호트 계산 로직 (제공해주신 코드 기반) ---
    valid_orders['order_month'] = valid_orders['created_at'].dt.to_period('M').dt.to_timestamp()
    
    first_purchase = valid_orders.groupby('user_id')['order_month'].min().rename('order_month_cohort')
    valid_orders = valid_orders.join(first_purchase, on='user_id')
    
    # 코호트 월이 없는 경우(join 실패) 데이터 제외
    valid_orders.dropna(subset=['order_month_cohort'], inplace=True)

    valid_orders['cohort_index'] = (
        (valid_orders['order_month'].dt.year - valid_orders['order_month_cohort'].dt.year) * 12 +
        (valid_orders['order_month'].dt.month - valid_orders['order_month_cohort'].dt.month)
    )
    
    cohort_pivot = valid_orders.groupby(['order_month_cohort', 'cohort_index'])['user_id'].nunique().reset_index()
    
    cohort_size = cohort_pivot[cohort_pivot['cohort_index'] == 0][['order_month_cohort', 'user_id']]
    cohort_pivot = cohort_pivot.merge(cohort_size, on='order_month_cohort', suffixes=('', '_cohort_size'))
    
    cohort_pivot['retention'] = cohort_pivot['user_id'] / cohort_pivot['user_id_cohort_size']

    cohort_table = cohort_pivot.pivot_table(index="order_month_cohort",
                                            columns="cohort_index",
                                            values="retention")
    

    # 1. 시각화할 데이터에서 첫 번째 열 (0개월차)을 제외합니다.
    # .iloc[:, 1:]는 모든 행과, 1번 인덱스(두 번째) 열부터 끝까지의 열을 선택합니다.
    heatmap_data = cohort_table.iloc[:, 1:]

    # 2. 주석(annot) 데이터도 동일하게 첫 번째 열을 제외하고 생성합니다.
    annot_data = None
    if show_annotations:
        annot_data = heatmap_data.copy()
        annot_data[annot_data == 0] = np.nan

    # --- Matplotlib 히트맵 생성 ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        data=heatmap_data,         # ✨ 수정: 슬라이싱된 데이터를 사용
        annot=annot_data,          # ✨ 수정: 슬라이싱된 주석 데이터를 사용
        fmt=".1%",
        cmap="Blues", 
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title("고객 리텐션 코호트 분석 히트맵", fontsize=18, fontweight='bold')
    ax.set_ylabel("첫 구매월 (Cohort)", fontsize=12)
    ax.set_xlabel("재구매까지 걸린 개월 수", fontsize=12)
    
    # Y축 날짜 포맷 변경
    ax.set_yticklabels([d.strftime('%Y-%m') for d in cohort_table.index])
    
    fig.tight_layout()
    
    return fig, cohort_table


def create_category_cohort_heatmap(order_items_df, products_df, category, show_annotations=True):
    """
    선택된 카테고리를 기준으로 코호트 리텐션 히트맵을 생성합니다.
    """
    valid_orders = order_items_df[order_items_df['status'] == 'Complete'].copy()

    orders_with_products = valid_orders.merge(
        products_df[['id', 'category']],
        left_on='product_id',
        right_on='id',
        how='left'
    )
    category_orders = orders_with_products[orders_with_products['category'] == category].copy()
    
    if category_orders.empty:
        return None, None
    
    category_orders['created_at'] = category_orders['created_at'].dt.tz_localize(None)
    category_orders['order_month'] = category_orders['created_at'].dt.to_period('M').dt.to_timestamp()
    
    first_purchase = category_orders.groupby('user_id')['order_month'].min().rename('order_month_cohort')
    category_orders = category_orders.join(first_purchase, on='user_id')
    category_orders.dropna(subset=['order_month_cohort'], inplace=True)
    
    # ✨ 수정: 연도 필터링 로직 제거
    # category_orders = category_orders[category_orders['order_month_cohort'].dt.year == year]

    if category_orders.empty:
        return None, None

    category_orders['cohort_index'] = ((category_orders['order_month'].dt.year - category_orders['order_month_cohort'].dt.year) * 12 + (category_orders['order_month'].dt.month - category_orders['order_month_cohort'].dt.month))
    
    cohort_pivot = category_orders.groupby(['order_month_cohort', 'cohort_index'])['user_id'].nunique().reset_index()
    cohort_size = cohort_pivot[cohort_pivot['cohort_index'] == 0][['order_month_cohort', 'user_id']]
    cohort_pivot = cohort_pivot.merge(cohort_size, on='order_month_cohort', suffixes=('', '_cohort_size'))
    cohort_pivot['retention'] = cohort_pivot['user_id'] / cohort_pivot['user_id_cohort_size']
    cohort_table = cohort_pivot.pivot_table(index="order_month_cohort", columns="cohort_index", values="retention")

    heatmap_data = cohort_table.iloc[:, 1:]
    
    annot_data = None
    if show_annotations:
        annot_data = heatmap_data.copy()
        annot_data[annot_data == 0] = np.nan

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(data=heatmap_data, annot=annot_data, fmt=".1%", cmap="Blues", linewidths=.5, ax=ax)
    
    ax.set_title(f"'{category}' 카테고리 고객 리텐션 히트맵", fontsize=18, fontweight='bold') # ✨ 수정: 제목에서 연도 제거
    ax.set_ylabel("첫 구매월 (Cohort)", fontsize=12)
    ax.set_xlabel("재구매까지 걸린 개월 수", fontsize=12)
    ax.set_yticklabels([d.strftime('%Y-%m') for d in heatmap_data.index])
    fig.tight_layout()
    
    return fig, cohort_table

@st.cache_data
# ✨ 수정: year 파라미터 제거
def create_advanced_cohort_heatmap(orders_df, max_age_m, show_annotations=True):
    """
    정교한 방식으로 코호트 재구매율을 계산하고 히트맵을 생성합니다.
    """
    # (앞부분 로직은 동일)
    use_cols = [c for c in ['user_id', 'created_at', 'status'] if c in orders_df.columns]
    if not use_cols or 'user_id' not in use_cols or 'created_at' not in use_cols:
        st.warning("분석에 필요한 'user_id', 'created_at' 컬럼이 데이터에 없습니다.")
        return None, None
    src = orders_df.loc[:, use_cols].copy()
    if 'status' in src.columns:
        src['status'] = src['status'].astype(str).str.strip().str.lower()
        src = src[src['status'] == 'complete']
    if src.empty:
        st.warning("상태가 'Complete'인 주문이 없습니다.")
        return None, None
    src = src[src['user_id'].notna()].copy()
    src['created_at'] = pd.to_datetime(src['created_at'], utc=True, errors='coerce')
    src = src.dropna(subset=['created_at'])
    if src.empty:
        st.warning("유효한 주문 시간이 있는 데이터가 없습니다.")
        return None, None
    last_month = src['created_at'].dt.to_period('M').max()
    last_i = last_month.year * 12 + last_month.month

    # 사용자별 첫 구매월 산출
    first = src.groupby('user_id', as_index=False)['created_at'].min().rename(columns={'created_at':'first_time'})
    first['cohort_month'] = first['first_time'].dt.to_period('M')
    
    # ✨ 수정: 연도 필터링 로직 제거 (모든 코호트 사용)
    # first_year = first[first['cohort_year'] == year].copy()
    # if first_year.empty: ...
        
    cohort_size = first.groupby('cohort_month')['user_id'].nunique().rename('cohort_size')

    # 주문에 코호트 라벨 붙이기
    lab = src.merge(first[['user_id','cohort_month']], on='user_id', how='inner')
    if lab.empty:
        st.warning("코호트 그룹의 주문 내역이 없습니다.")
        return None, None
    lab['order_month'] = lab['created_at'].dt.to_period('M')

    # (이하 계산 로직은 이전과 동일)
    cm_i = lab['cohort_month'].dt.year * 12 + lab['cohort_month'].dt.month
    om_i = lab['order_month'].dt.year * 12 + lab['order_month'].dt.month
    lab['cohort_age_m'] = om_i - cm_i
    lab = lab[(lab['cohort_age_m'] >= 0) & (lab['cohort_age_m'] <= max_age_m)].copy()

    cohort_months = np.sort(first['cohort_month'].unique()) # ✨ 수정: first_year -> first
    age_vals = np.arange(0, max_age_m + 1)
    grid = pd.MultiIndex.from_product([cohort_months, age_vals], names=['cohort_month','cohort_age_m']).to_frame(index=False)
    grid['cm_i'] = grid['cohort_month'].dt.year * 12 + grid['cohort_month'].dt.month
    grid['order_i'] = grid['cm_i'] + grid['cohort_age_m']
    grid = grid[grid['order_i'] <= last_i].drop(columns=['cm_i','order_i'])

    counts = lab.groupby(['cohort_month','cohort_age_m'])['user_id'].nunique().rename('active_users').reset_index()
    counts = grid.merge(counts, on=['cohort_month','cohort_age_m'], how='left').fillna({'active_users': 0})

    counts = counts.merge(cohort_size, on='cohort_month', how='left')
    counts['retention_rate'] = counts['active_users'] / counts['cohort_size']

    heat = counts.pivot(index='cohort_month', columns='cohort_age_m', values='retention_rate')
    heat = heat.sort_index().sort_index(axis=1)
    if 0 in heat.columns:
        heat = heat.loc[:, heat.columns != 0]

    idx_period = heat.index.copy()
    base_labels = idx_period.astype(str)
    n_map = cohort_size.reindex(idx_period).fillna(0).astype(int).map(lambda x: f"{x:,}")
    row_labels = base_labels + ' · N=' + n_map
    heat.index = row_labels
    heat_pct = heat * 100

    # 히트맵 시각화
    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(heat_pct.index))))
    sns.heatmap(
        heat_pct, annot=show_annotations, fmt=".1f", cmap="Blues",
        cbar_kws={'label': '재구매율 (%)'}, linewidths=.3, linecolor='white', ax=ax
    )
    ax.set_title("월별 코호트 재구매율 히트맵 (Age≥1)", fontsize=16) # ✨ 수정: 제목에서 연도 제거
    ax.set_xlabel("첫 구매 후 경과 개월 수")
    ax.set_ylabel("코호트 월 (첫 구매월 · N=표본크기)")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    return fig, heat

def create_repeat_purchaser_chart(orders_df):
    """
    2023년 월별 재구매자 비율을 분석하고 이중 축 그래프를 생성합니다.
    """
    # (앞부분 데이터 처리 로직은 동일)
    use_cols = [c for c in ['user_id', 'created_at', 'status'] if c in orders_df.columns]
    orders = orders_df.loc[:, use_cols].copy()
    orders = orders[orders['user_id'].notna()].copy()
    orders['created_at'] = pd.to_datetime(orders['created_at'], utc=True, errors='coerce')
    orders = orders.dropna(subset=['created_at'])
    if 'status' in orders.columns:
        orders['status'] = orders['status'].astype(str).str.strip().str.lower()
        orders = orders[orders['status'] == 'complete'].drop(columns=['status'])
    if orders.empty:
        st.warning("상태가 'Complete'인 주문이 없습니다.")
        return None, None

    first_time = orders.groupby('user_id', as_index=False)['created_at'].min().rename(columns={'created_at': 'first_time'})
    first_time['cohort_month'] = first_time['first_time'].dt.to_period('M')
    purch = orders[['user_id','created_at']].copy()
    purch['order_month'] = purch['created_at'].dt.to_period('M')
    purch = purch.drop_duplicates(['user_id','order_month'])
    purch = purch.merge(first_time[['user_id','cohort_month']], on='user_id', how='left')
    purch['is_returning'] = purch['cohort_month'] < purch['order_month']
    by_month = purch.groupby('order_month')['is_returning'].agg(returning_users='sum', purchasers='count').sort_index()
    by_month['rate_raw'] = by_month['returning_users'] / by_month['purchasers']
    by_month['repeat_purchaser_rate'] = by_month['rate_raw'].round(3)

    # ✨ 수정: 2023년으로 연도 고정
    m2023 = by_month.loc[by_month.index.year == 2023].copy()
    if m2023.empty:
        st.warning("2023년 데이터가 없습니다.")
        return None, None
    m2023.index = m2023.index.astype(str)

    # (이하 시각화 코드는 이전과 동일)
    fig, ax1 = plt.subplots(figsize=(13, 5))
    x = np.arange(len(m2023))
    months = m2023.index.to_list()
    bars = ax1.bar(x, m2023['purchasers'], width=0.6, color='skyblue', alpha=0.9, label='총 구매자 수')
    ax1.set_ylabel('총 구매자 수')
    ax1.set_xlabel('주문월 (YYYY-MM)')
    ax1.set_xticks(x); ax1.set_xticklabels(months, rotation=45, ha='right')
    ax2 = ax1.twinx()
    rate_pct = m2023['rate_raw'] * 100.0
    ax2.plot(x, rate_pct, marker='o', linewidth=2, color='tab:blue', markerfacecolor='white', label='재구매자 비율')
    ax2.set_ylabel('재구매자 비율 (%)')
    ax2.set_ylim(0, max(5, np.ceil(rate_pct.max()/5)*5))
    for rect, n in zip(bars, m2023['purchasers']):
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height()*0.5, f"{int(n):,}", ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    for xi, rp in zip(x, rate_pct):
        ax2.annotate(f"{rp:.1f}%", xy=(xi, rp), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=9, color='tab:blue', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    is_nov_dec = [m.endswith(('-11','-12')) for m in months]
    is_jan_oct = [not f for f in is_nov_dec]
    if any(is_jan_oct):
        early_ret = m2023.loc[is_jan_oct, 'returning_users'].sum()
        early_tot = m2023.loc[is_jan_oct, 'purchasers'].sum()
        early_avg_w = (early_ret / early_tot) * 100.0
        ax2.hlines(early_avg_w, -0.5, len(x)-0.5, colors='gray', linestyles='dashed', linewidth=1.5, label='1-10월 평균 (가중)')
    if any(is_nov_dec):
        late_ret  = m2023.loc[is_nov_dec, 'returning_users'].sum()
        late_tot  = m2023.loc[is_nov_dec, 'purchasers'].sum()
        late_avg_w  = (late_ret / late_tot)  * 100.0
        ax2.hlines(late_avg_w, -0.5, len(x)-0.5, colors='orange', linestyles='dashed', linewidth=1.5, label='11-12월 평균 (가중)')
    fig.suptitle("2023년 월별 재구매자 비율", fontsize=16, fontweight='bold') # ✨ 수정
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.055, 1.0))
    fig.tight_layout(rect=[0,0,0.86,1])

    return fig, m2023

@st.cache_data
def create_weekly_cohort_heatmap(orders_df, selected_month, max_age_w, show_annotations=True):
    """
    선택된 월에 시작된 주간 코호트의 재구매율을 분석하고 히트맵을 생성합니다.
    """
    # --- 제공해주신 코드 로직을 스트림릿 함수에 맞게 수정 ---
    
    # 0) 원천 정리
    use_cols = [c for c in ['user_id', 'created_at', 'status'] if c in orders_df.columns]
    src = orders_df.loc[:, use_cols].copy()
    src = src[src['user_id'].notna()].copy()
    src['created_at'] = pd.to_datetime(src['created_at'], utc=True, errors='coerce')
    src = src.dropna(subset=['created_at']).sort_values(['user_id','created_at'])
    if 'status' in src.columns:
        src['status'] = src['status'].astype(str).str.strip().str.lower()
        src = src[src['status'] == 'complete']
    if src.empty:
        st.warning("상태가 'Complete'인 주문이 없습니다.")
        return None, None

    REF = pd.Timestamp('1970-01-05', tz='UTC')
    src['week_start'] = src['created_at'].dt.normalize() - pd.to_timedelta(src['created_at'].dt.dayofweek, unit='D')
    src['week_idx'] = ((src['week_start'] - REF).dt.days // 7).astype(int)
    last_week_idx = int(src['week_idx'].max())

    # 1) 첫 구매(코호트) 계산
    first = src.groupby('user_id', as_index=False)['created_at'].min().rename(columns={'created_at':'first_time'})
    first['cohort_week_start'] = first['first_time'].dt.normalize() - pd.to_timedelta(first['first_time'].dt.dayofweek, unit='D')
    first['cohort_week_idx'] = ((first['cohort_week_start'] - REF).dt.days // 7).astype(int)
    first['cohort_month'] = first['cohort_week_start'].dt.to_period('M').astype(str)

    # ✨ 수정: 선택된 월의 코호트만 필터링
    cohorts_in_month = first[first['cohort_month'] == selected_month].copy()
    if cohorts_in_month.empty:
        st.warning(f"{selected_month}에 시작된 코호트 그룹이 없습니다.")
        return None, None

    # 라벨 생성
    iso = cohorts_in_month['cohort_week_start'].dt.isocalendar()
    cohorts_in_month['cohort_week_lbl'] = iso['year'].astype(str) + '-W' + iso['week'].astype(str).str.zfill(2)
    cohorts_in_month['week_of_month'] = ((cohorts_in_month['cohort_week_start'].dt.day - 1) // 7 + 1).astype(int)
    cohorts_in_month['cohort_mweek_lbl'] = (cohorts_in_month['cohort_month'] + ' W' + cohorts_in_month['week_of_month'].astype(str) + ' (' + cohorts_in_month['cohort_week_lbl'] + ')')
    
    cohort_size = cohorts_in_month.groupby('cohort_week_idx')['user_id'].nunique().rename('cohort_size')

    # (이하 계산 로직은 이전과 동일하나, 'cohorts_in_month'를 사용)
    lab = src.merge(cohorts_in_month[['user_id','cohort_week_idx']], on='user_id', how='inner')
    if lab.empty: return None, None
    lab['age_w'] = lab['week_idx'] - lab['cohort_week_idx']
    lab = lab[(lab['age_w'] >= 0) & (lab['age_w'] <= max_age_w)].copy()
    cohort_weeks = np.sort(cohorts_in_month['cohort_week_idx'].unique())
    age_vals = np.arange(0, max_age_w+1)
    grid = pd.MultiIndex.from_product([cohort_weeks, age_vals], names=['cohort_week_idx','age_w']).to_frame(index=False)
    grid = grid[(grid['cohort_week_idx'] + grid['age_w']) <= last_week_idx].copy()
    counts = lab.groupby(['cohort_week_idx','age_w'])['user_id'].nunique().rename('active_users').reset_index()
    counts = grid.merge(counts, on=['cohort_week_idx','age_w'], how='left').fillna({'active_users':0})
    counts = counts.merge(cohort_size, on='cohort_week_idx', how='left')
    counts['retention_rate'] = counts['active_users'] / counts['cohort_size']
    heat = counts.pivot(index='cohort_week_idx', columns='age_w', values='retention_rate').sort_index().sort_index(axis=1)
    if 0 in heat.columns:
        heat = heat.loc[:, heat.columns != 0]
    lbl_map = (cohorts_in_month[['cohort_week_idx','cohort_mweek_lbl']].drop_duplicates().set_index('cohort_week_idx')['cohort_mweek_lbl'])
    cohort_idx = heat.index.copy()
    base_labels = cohort_idx.map(lbl_map)
    n_map = cohort_size.reindex(cohort_idx).fillna(0).astype(int).map(lambda x: f"{x:,}")
    row_labels = base_labels + ' · N=' + n_map
    heat.index = row_labels
    heat_pct = heat * 100

    # 히트맵 시각화
    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(heat_pct))))
    sns.heatmap(
        heat_pct, annot=show_annotations, fmt=".1f", cmap="Blues",
        cbar_kws={'label':'재구매율 (%)'}, linewidths=.3, linecolor='white', ax=ax)
    ax.set_title(f"{selected_month} 시작 주간 코호트 재구매율 (Age≥1)", fontsize=16)
    ax.set_xlabel("첫 구매 후 경과 주 수")
    ax.set_ylabel("코호트 주 (YYYY-MM Wn (ISO 주) · N)")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    return fig, heat

@st.cache_data
def create_daily_cohort_heatmap(orders_df, selected_month, selected_week, max_age_d, show_annotations=True):
    """
    일 단위 코호트 재구매율을 계산하고 월/주 필터를 적용하여 히트맵을 생성합니다.
    """
    # --- 제공해주신 코드 로직을 스트림릿 함수에 맞게 수정 ---
    
    # 0) 원천 정리
    use_cols = [c for c in ['user_id', 'created_at', 'status'] if c in orders_df.columns]
    src = orders_df.loc[:, use_cols].copy()
    src = src[src['user_id'].notna()]
    src['status'] = src['status'].astype(str).str.strip().str.lower()
    src = src[src['status'] == 'complete'].drop(columns=['status'])
    if src.empty:
        st.warning("상태가 'Complete'인 주문이 없습니다.")
        return None, None
    src['created_at'] = pd.to_datetime(src['created_at'], utc=True, errors='coerce')
    src = src.dropna(subset=['created_at']).sort_values(['user_id','created_at'])
    if src.empty:
        st.warning("유효한 주문 시간이 있는 데이터가 없습니다.")
        return None, None
    src['order_day'] = src['created_at'].dt.floor('D')
    last_day = src['order_day'].max()

    # 1) 코호트(첫 구매 일자) 계산
    first = src.groupby('user_id', as_index=False)['order_day'].min().rename(columns={'order_day':'cohort_day'})
    first['cohort_month'] = first['cohort_day'].dt.to_period('M').astype(str)
    
    # 주차(Week of Month) 계산
    week_start_day = first['cohort_day'].dt.normalize() - pd.to_timedelta(first['cohort_day'].dt.dayofweek, unit='D')
    first['cohort_week_of_month'] = ((week_start_day.dt.day - 1) // 7 + 1)

    # ✨ 수정: 선택된 월/주로 코호트 필터링
    cohorts_filtered = first[first['cohort_month'] == selected_month]
    if selected_week != 'All':
        cohorts_filtered = cohorts_filtered[cohorts_filtered['cohort_week_of_month'] == selected_week]
    
    if cohorts_filtered.empty:
        st.warning(f"선택된 조건에 맞는 코호트 그룹이 없습니다.")
        return None, None
        
    cohort_size = cohorts_filtered.groupby('cohort_day')['user_id'].nunique().rename('cohort_size')

    # (이하 계산 로직은 이전과 동일하나, 'cohorts_filtered'를 사용)
    lab = src.merge(cohorts_filtered[['user_id','cohort_day']], on='user_id', how='inner')
    lab['age_d'] = (lab['order_day'] - lab['cohort_day']).dt.days
    lab = lab[(lab['age_d'] >= 0) & (lab['age_d'] <= max_age_d)].copy()
    cohort_days = np.sort(cohorts_filtered['cohort_day'].unique())
    age_vals = np.arange(0, max_age_d+1)
    grid = pd.MultiIndex.from_product([cohort_days, age_vals], names=['cohort_day','age_d']).to_frame(index=False)
    grid['order_day'] = grid['cohort_day'] + pd.to_timedelta(grid['age_d'], unit='D')
    grid = grid[grid['order_day'] <= last_day].drop(columns=['order_day'])
    counts = lab.groupby(['cohort_day','age_d'])['user_id'].nunique().rename('active_users').reset_index()
    counts = grid.merge(counts, on=['cohort_day','age_d'], how='left').fillna({'active_users':0})
    counts = counts.merge(cohort_size, on='cohort_day', how='left')
    counts['retention_rate'] = counts['active_users'] / counts['cohort_size']

    # 시각화 데이터 준비 (Age=0 제외)
    min_age_d = 1
    col_range = list(range(min_age_d, max_age_d + 1))
    heat = counts.pivot(index='cohort_day', columns='age_d', values='retention_rate').sort_index().reindex(columns=col_range)
    if heat.empty:
        st.warning("선택된 조건의 재구매 데이터가 없습니다.")
        return None, None
        
    # 라벨 생성
    def make_row_labels(days_index: pd.Index) -> pd.Series:
        idx = pd.Series(days_index, index=days_index)
        week_start = idx.dt.normalize() - pd.to_timedelta(idx.dt.dayofweek, unit='D')
        wom = ((week_start.dt.day - 1) // 7 + 1)
        dow_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
        dow_eng = idx.dt.dayofweek.map(dow_map)
        date_str = idx.dt.date.astype(str)
        return date_str + ' (' + dow_eng + ', W' + wom.astype(str) + ')'
        
    idx_dt = heat.index
    base_lbl = make_row_labels(idx_dt)
    n_map = cohort_size.reindex(idx_dt).fillna(0).astype(int).map(lambda x: f"{x:,}")
    heat.index = base_lbl + ' · N=' + n_map
    heat_pct = heat * 100

    # --- ✨✨ 수정된 시각화 부분 ✨✨ ---

    # Figure 높이 동적 조절 (행당 할당 높이를 1.2로 대폭 증가)
    base_height = 8
    row_height_factor = 1.2  # ✨ 수정: 각 행의 높이를 더욱 확보
    fig_h = max(base_height, len(heat_pct.index) * row_height_factor)
    
    # Figure 너비 동적 조절
    base_width = 12
    col_width_factor = 0.9
    fig_w = max(base_width, len(heat_pct.columns) * col_width_factor)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    month_max = heat.max().max() if not heat.empty else 0.01
    vmax = float(month_max * 100.0) if month_max > 0 else 1.0

    sns.heatmap(
        heat_pct, 
        annot=show_annotations, 
        fmt=".1f",
        cmap="Blues",
        vmin=0, 
        vmax=vmax,
        linewidths=.5, # 선 굵기
        linecolor='white',
        cbar_kws={'label': '재구매율 (%)'}, 
        ax=ax,
        # ✨ 수정: annot_kws를 사용하여 annotation 글꼴 크기를 14로 대폭 증가
        annot_kws={"fontsize": 20} 
    )
    ax.set_title(f"{selected_month} 일일 코호트 재구매율 (Age≥1)", fontsize=18)
    ax.set_xlabel(f"첫 구매 후 경과 일 수 ({min_age_d}–{max_age_d})", fontsize=30)
    ax.set_ylabel("코호트", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=20) # 축 틱 라벨 크기 일괄 조절
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    return fig, heat


# --- ✨ [함수 추가] 요일별 재구매 분석 ---

# --- Helper 함수들 (분석 함수 내부에 포함시키거나 전역으로 둠) ---
def fmt_pct3(x):
    if pd.isna(x): return "NA"
    return f"{x*100:.3f}%"

def agg_weekday(series_wd, df):
    g = (df.groupby(series_wd, as_index=False).agg(Repeat_Orders=('active_users','sum'), Exposure=('cohort_size','sum')))
    g = (g.set_index(series_wd.name).reindex(range(7)).fillna(0.0).reset_index().rename(columns={'index': series_wd.name}))
    g['Repeat_Rate'] = np.where(g['Exposure'] > 0, g['Repeat_Orders'] / g['Exposure'], np.nan)
    g['Weekday'] = g[series_wd.name].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
    return g.sort_values(series_wd.name).reset_index(drop=True)

def set_padded_ylim(ax, *series, low_pad=0.15, high_pad=0.25, min_range=1e-6):
    vals = np.concatenate([np.array([v for v in s if pd.notna(v)]) for s in series if len(s)])
    if vals.size == 0: return
    y_min, y_max = float(vals.min()), float(vals.max())
    rng = max(y_max - y_min, min_range)
    ax.set_ylim(max(0.0, y_min - low_pad*rng), y_max + high_pad*rng)

@st.cache_data
def create_weekday_repeat_purchase_charts(orders_df, start_date, end_date):
    """
    요일별 재구매 패턴을 분석하고 3개의 차트를 포함한 Figure를 생성합니다.
    """
    # --- 데이터 준비 ---
    # (이전 고급 코호트 분석에서 사용한 'counts' 데이터프레임을 생성하는 로직을 재활용)
    src = orders_df[orders_df['status'].str.lower() == 'complete'].copy()
    src['created_at'] = pd.to_datetime(src['created_at'], utc=True, errors='coerce')
    src = src.dropna(subset=['created_at']).sort_values(['user_id','created_at'])
    start_datetime = pd.to_datetime(start_date).tz_localize('UTC')
    end_datetime = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
    src = src[(src['created_at'] >= start_datetime) & (src['created_at'] < end_datetime)]
    if src.empty:
        st.warning("선택된 기간에 'Complete' 상태의 주문이 없습니다.")
        return None, None, None
        
    src['order_day'] = src['created_at'].dt.floor('D')
    first = src.groupby('user_id', as_index=False)['order_day'].min().rename(columns={'order_day':'cohort_day'})
    lab = src.merge(first[['user_id','cohort_day']], on='user_id', how='inner')
    lab['age_d'] = (lab['order_day'] - lab['cohort_day']).dt.days
    cohort_size = first.groupby('cohort_day')['user_id'].nunique().rename('cohort_size')
    counts = lab.groupby(['cohort_day','age_d'])['user_id'].nunique().rename('active_users').reset_index()
    counts = counts.merge(cohort_size, on='cohort_day', how='left')
    
    # --- 제공해주신 코드 로직 (Prep, Aggregations) ---
    df = counts.copy()
    df = df[df['age_d'] >= 1].copy()
    if df.empty:
        st.warning("재구매 데이터(Age≥1)가 없습니다.")
        return None, None, None
        
    df['cohort_day'] = pd.to_datetime(df['cohort_day'], utc=True, errors='coerce')
    df = df.dropna(subset=['cohort_day'])
    df['repurch_date'] = df['cohort_day'] + pd.to_timedelta(df['age_d'], unit='D')
    df['order_wd'] = df['repurch_date'].dt.dayofweek
    df['cohort_wd'] = df['cohort_day'].dt.dayofweek
    
    order_grp = agg_weekday(df['order_wd'], df)
    cohort_grp = agg_weekday(df['cohort_wd'], df)
    
    # --- 시각화 ---
    fig, (ax_bars_ord, ax_bars_coh, ax_line_both) = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    x = np.arange(7)
    week_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    # Plot 1: Order bars
    bars1 = ax_bars_ord.bar(x, order_grp['Repeat_Orders'], color='#4C78A8', alpha=0.9)
    ax_bars_ord.set(title="재구매 발생 요일 분포", ylabel="재구매 건수", xticks=x, xticklabels=week_labels)
    ax_bars_ord.grid(axis='y', linestyle=':', alpha=0.5)

    # Plot 2: Cohort bars
    bars2 = ax_bars_coh.bar(x, cohort_grp['Repeat_Orders'], color='#E45756', alpha=0.9)
    ax_bars_coh.set(title="첫 구매 요일별 재구매 건수", ylabel="재구매 건수", xticks=x, xticklabels=week_labels)
    ax_bars_coh.grid(axis='y', linestyle=':', alpha=0.5)

    # Plot 3: Combined lines
    order_line, = ax_line_both.plot(x, order_grp['Repeat_Rate'], marker='o', color='#4C78A8', label='재구매일 기준')
    cohort_line, = ax_line_both.plot(x, cohort_grp['Repeat_Rate'], marker='o', color='#E45756', label='첫구매일 기준')
    ax_line_both.set(title="요일별 재구매율 (재구매일 vs 첫구매일)", ylabel="재구매율 (%)", xticks=x, xticklabels=week_labels)
    ax_line_both.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax_line_both.grid(axis='y', linestyle=':', alpha=0.5)
    set_padded_ylim(ax_line_both, order_grp['Repeat_Rate'].values, cohort_grp['Repeat_Rate'].values)
    ax_line_both.legend(loc='best')
    
    fig.suptitle("요일별 재구매 패턴 분석 (재구매 건수 및 비율, Age≥1)", fontsize=16, fontweight='bold')
    
    return fig, order_grp, cohort_grp


# Helper 함수들
def fmt_pct3(x):
    if pd.isna(x): return "NA"
    return f"{x*100:.3f}%"

@st.cache_data
def create_weekday_weekend_chart(orders_df, start_date, end_date):
    """
    선택된 기간의 데이터를 기반으로 주중/주말 재구매 패턴을 분석하고 시각화합니다.
    """
    # --- 데이터 준비 (기존 코호트 분석 로직 재활용하여 'counts' 생성) ---
    src = orders_df[orders_df['status'].str.lower() == 'complete'].copy()
    src['created_at'] = pd.to_datetime(src['created_at'], utc=True, errors='coerce')
    src = src.dropna(subset=['created_at']).sort_values(['user_id','created_at'])
    start_datetime = pd.to_datetime(start_date).tz_localize('UTC')
    end_datetime = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
    src = src[(src['created_at'] >= start_datetime) & (src['created_at'] < end_datetime)]
    if src.empty:
        st.warning("선택된 기간에 'Complete' 상태의 주문이 없습니다.")
        return None, None
        
    src['order_day'] = src['created_at'].dt.floor('D')
    first = src.groupby('user_id', as_index=False)['order_day'].min().rename(columns={'order_day':'cohort_day'})
    lab = src.merge(first[['user_id','cohort_day']], on='user_id', how='inner')
    lab['age_d'] = (lab['order_day'] - lab['cohort_day']).dt.days
    cohort_size = first.groupby('cohort_day')['user_id'].nunique().rename('cohort_size')
    counts = lab.groupby(['cohort_day','age_d'])['user_id'].nunique().rename('active_users').reset_index()
    counts = counts.merge(cohort_size, on='cohort_day', how='left')
    
    # --- 제공해주신 코드 로직 (Prep, Aggregations) ---
    df = counts.copy()
    df = df[df['age_d'] >= 1].copy()
    if df.empty:
        st.warning("재구매 데이터(Age≥1)가 없습니다.")
        return None, None
        
    df['cohort_day'] = pd.to_datetime(df['cohort_day'], utc=True, errors='coerce')
    df = df.dropna(subset=['cohort_day'])
    df['repurch_date'] = df['cohort_day'] + pd.to_timedelta(df['age_d'], unit='D')
    df['repurch_wd'] = df['repurch_date'].dt.dayofweek
    df['is_weekend'] = df['repurch_wd'].isin({5,6})

    g = (df.groupby('is_weekend', as_index=False)
           .agg(Repeaters=('active_users','sum'), Exposure=('cohort_size','sum')))
    g['Rate'] = np.where(g['Exposure']>0, g['Repeaters']/g['Exposure'], np.nan)
    g['Group'] = np.where(g['is_weekend'], '주말 (토+일)', '주중 (월–금)')
    g = g.sort_values('is_weekend').reset_index(drop=True)

    # 테이블 생성
    tbl = pd.DataFrame({
        '구분': g['Group'],
        '재구매자 수': g['Repeaters'].astype(int),
        '전체 코호트 크기': g['Exposure'].astype(int),
        '재구매율 (%)': (g['Rate']*100).round(3)
    })

    # 막대 차트 생성
    fig, ax = plt.subplots(figsize=(7, 5))
    BAR_COLORS = ['#A3C4F3', '#FDE2A7']
    bars = ax.bar(g['Group'], g['Rate'], color=BAR_COLORS, edgecolor='none')
    
    ymax = float(np.nanmax(g['Rate'])) if len(g) else 0.0
    pad = ymax * 0.15 if ymax > 0 else 0.01
    ax.set_ylim(0, ymax + pad)

    for rect, r in zip(bars, g['Rate']):
        ax.annotate(f"{r*100:.2f}%", xy=(rect.get_x() + rect.get_width()/2, r),
                    xytext=(0, 6), textcoords='offset points', ha='center', va='bottom')
                    
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylabel("재구매율 (%)")
    ax.set_title("주중 vs 주말 재구매율 비교 (Age ≥ 1)", fontsize=14)
    ax.grid(axis='y', linestyle=':', alpha=0.35)
    fig.tight_layout()

    return fig, tbl


@st.cache_data
def create_weekly_cohort_heatmap(orders_df, selected_month, selected_week, max_age_w, show_annotations=True):
    """
    선택된 월/주에 시작된 주간 코호트의 재구매율을 분석하고 히트맵을 생성합니다.
    """
    # --- 제공해주신 코드 로직을 스트림릿 함수에 맞게 수정 ---
    
    # 0) 원천 정리
    use_cols = [c for c in ['user_id', 'created_at', 'status'] if c in orders_df.columns]
    src = orders_df.loc[:, use_cols].copy()
    src = src[src['user_id'].notna()]
    src['status'] = src['status'].astype(str).str.strip().str.lower()
    src = src[src['status'] == 'complete'].drop(columns=['status'])
    if src.empty:
        st.warning("상태가 'Complete'인 주문이 없습니다.")
        return None, None
    src['created_at'] = pd.to_datetime(src['created_at'], utc=True, errors='coerce')
    src = src.dropna(subset=['created_at']).sort_values(['user_id','created_at'])
    if src.empty:
        st.warning("유효한 주문 시간이 있는 데이터가 없습니다.")
        return None, None
        
    REF = pd.Timestamp('1970-01-05', tz='UTC')
    src['week_start'] = src['created_at'].dt.normalize() - pd.to_timedelta(src['created_at'].dt.dayofweek, unit='D')
    src['week_idx'] = ((src['week_start'] - REF).dt.days // 7).astype(int)
    last_week_idx = int(src['week_idx'].max())

    # 1) 첫 구매(코호트) 계산 및 라벨 생성
    first = src.groupby('user_id', as_index=False)['created_at'].min().rename(columns={'created_at':'first_time'})
    first['cohort_week_start'] = first['first_time'].dt.normalize() - pd.to_timedelta(first['first_time'].dt.dayofweek, unit='D')
    first['cohort_week_idx'] = ((first['cohort_week_start'] - REF).dt.days // 7).astype(int)
    iso = first['cohort_week_start'].dt.isocalendar()
    first['cohort_year'] = iso['year']
    first['cohort_week_lbl'] = iso['year'].astype(str) + '-W' + iso['week'].astype(str).str.zfill(2)
    first['cohort_month'] = first['cohort_week_start'].dt.to_period('M').astype(str)
    first['week_of_month'] = ((first['cohort_week_start'].dt.day - 1) // 7 + 1).astype(int)
    first['cohort_mweek_lbl'] = (first['cohort_month'] + ' W' + first['week_of_month'].astype(str) + ' (' + first['cohort_week_lbl'] + ')')
    
    # ✨ 수정: 선택된 월/주로 코호트 필터링
    cohorts_filtered = first[first['cohort_month'] == selected_month]
    if selected_week != 'All':
        cohorts_filtered = cohorts_filtered[cohorts_filtered['week_of_month'] == selected_week]
    
    if cohorts_filtered.empty:
        st.warning(f"선택된 조건에 맞는 코호트 그룹이 없습니다.")
        return None, None
        
    cohort_size = cohorts_filtered.groupby('cohort_week_idx')['user_id'].nunique().rename('cohort_size')

    # (이하 계산 로직은 이전과 동일하나, 'cohorts_filtered'를 사용)
    lab = src.merge(cohorts_filtered[['user_id','cohort_week_idx']], on='user_id', how='inner')
    if lab.empty: return None, None
    lab['age_w'] = lab['week_idx'] - lab['cohort_week_idx']
    lab = lab[(lab['age_w'] >= 0) & (lab['age_w'] <= max_age_w)].copy()
    cohort_weeks = np.sort(cohorts_filtered['cohort_week_idx'].unique())
    age_vals = np.arange(0, max_age_w+1)
    grid = pd.MultiIndex.from_product([cohort_weeks, age_vals], names=['cohort_week_idx','age_w']).to_frame(index=False)
    grid = grid[(grid['cohort_week_idx'] + grid['age_w']) <= last_week_idx].copy()
    counts = lab.groupby(['cohort_week_idx','age_w'])['user_id'].nunique().rename('active_users').reset_index()
    counts = grid.merge(counts, on=['cohort_week_idx','age_w'], how='left').fillna({'active_users':0})
    counts = counts.merge(cohort_size, on='cohort_week_idx', how='left')
    counts['retention_rate'] = counts['active_users'] / counts['cohort_size']
    heat = counts.pivot(index='cohort_week_idx', columns='age_w', values='retention_rate').sort_index().sort_index(axis=1)
    if 0 in heat.columns:
        heat = heat.loc[:, heat.columns != 0]

    lbl_map = (cohorts_filtered[['cohort_week_idx','cohort_mweek_lbl']].drop_duplicates().set_index('cohort_week_idx')['cohort_mweek_lbl'])
    cohort_idx = heat.index.copy()
    base_labels = cohort_idx.map(lbl_map)
    n_map = cohort_size.reindex(cohort_idx).fillna(0).astype(int).map(lambda x: f"{x:,}")
    row_labels = base_labels + ' · N=' + n_map
    heat.index = row_labels
    heat_pct = heat * 100

    # 히트맵 시각화
    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(heat_pct))))
    sns.heatmap(
        heat_pct, annot=show_annotations, fmt=".1f", cmap="Blues",
        cbar_kws={'label':'재구매율 (%)'}, linewidths=.3, linecolor='white', ax=ax)
    ax.set_title(f"{selected_month} (W{selected_week if selected_week != 'All' else '전체'}) 주간 코호트 재구매율", fontsize=16)
    ax.set_xlabel("첫 구매 후 경과 주 수")
    ax.set_ylabel("코호트 주 (YYYY-MM Wn (ISO 주) · N)")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    return fig, heat



# --- 🎨 메인 대시보드 레이아웃 ---

st.set_page_config(layout="wide", page_title="구매 분석")
st.title("🛍️ 사용자 구매 패턴 분석")

# --- 데이터 로딩 ---
all_data = load_all_data()

if not all_data :
    st.error("주문(order_items) 데이터를 불러오는데 실패했습니다.")
else:

    events = all_data["events"]
    order_items = all_data["order_items"]
    users = all_data["users"]
    orders = all_data["orders"]
    products = all_data["products"]
    products_master = products.copy()


    events_master = events.copy()
    order_items_master = order_items.copy()
    users_master = users.copy()
    orders_master = orders.copy()

    raw_data_schema={
            'user_id': 'session_id', 'event_name': 'event_type', 'event_timestamp': 'created_at'
        }
    
    valid_status = ['Complete', 'Returned', 'Cancelled']
    valid_orders = order_items[order_items['status'].isin(valid_status)].copy()

    # 유저별 구매 내역 정렬
    valid_orders = valid_orders.sort_values(['user_id', 'created_at'])
    

    valid_status = ['Complete', 'Returned', 'Cancelled']
    valid_orders = order_items[order_items['status'].isin(valid_status)].copy()
    user_purchase_counts = valid_orders.groupby('user_id')['order_id'].nunique()

    st.header("사용자별 구매 횟수 분포")
    st.write("각 사용자가 몇 번의 구매를 했는지 분포를 통해 충성 고객과 일회성 고객의 비율을 파악할 수 있습니다.")

    order_items_master = all_data["order_items"].copy()
    products_master = all_data["products"].copy()

    # --- 🎨 메인 대시보드 레이아웃 ---
    st.title("고객 리텐션 분석 (Cohort)")

            # --- 사이드바: 컨트롤 패널 ---
    st.sidebar.header("⚙️ 컨트롤 패널")
    st.sidebar.subheader("📅 날짜 필터")
    start_date = st.sidebar.date_input("시작일", order_items_master['created_at'].min())
    end_date = st.sidebar.date_input("종료일", order_items_master['created_at'].max())
            
    st.sidebar.divider()
    st.sidebar.subheader("📊 차트 옵션")
    # max_age_option = st.sidebar.slider("최대 경과 개월 수:", 1, 24, 12, help="히트맵에 표시할 최대 재구매 경과 개월 수를 선택합니다.")
    show_annotations = st.sidebar.checkbox("히트맵에 리텐션 값(%) 표시", value=True)

    tab1, tab2, tab3  = st.tabs(["🎯 사용자별 구매 횟수 분포","📈 월별 첫 구매 고객 재구매율 분석", "🌊 요일에 따른 재구매 패턴 심층 분석"])


    with tab1:
        st.subheader("전체 유입 경로 분포")

        if not all_data or "order_items" not in all_data:
            st.error("주문(order_items) 데이터를 불러오는데 실패했습니다.")
        else:
            order_items = all_data["order_items"]
            
            st.header("사용자별 구매 횟수 분포")
            st.write("각 사용자가 몇 번의 구매를 했는지 분포를 통해 충성 고객과 일회성 고객의 비율을 파악할 수 있습니다.")

            # 차트 생성 함수 호출
            dist_fig, dist_data = create_purchase_distribution_chart(order_items)
            
            if dist_fig:
                # 컬럼을 사용해 차트와 데이터를 나란히 표시
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.pyplot(dist_fig)
                with col2:
                    st.write("#### 데이터 요약")
                    st.dataframe(dist_data)

        if not all_data:
            st.error("데이터를 불러오는데 실패했습니다.")
        else:
            

            # --- ✨✨ 핵심 수정 부분 ✨✨ ---

            # --- 첫 번째 분석: 전체 리텐션 ---
            st.subheader("🗓️ 전체 고객 리텐션")
            
            # 날짜 필터링
            start_datetime = pd.to_datetime(start_date).tz_localize('UTC')
            end_datetime = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
            filtered_orders = order_items_master[
                (order_items_master['created_at'] >= start_datetime) & 
                (order_items_master['created_at'] < end_datetime)
            ]

            # 전체 리텐션 차트 생성 및 표시
            heatmap_fig, cohort_df = create_retention_heatmap(filtered_orders, show_annotations)
            if heatmap_fig:
                st.pyplot(heatmap_fig)
                with st.expander("상세 데이터 보기"):
                    st.dataframe(cohort_df.style.format("{:.2%}"))
            else:
                st.warning("선택된 기간에 분석할 데이터가 없습니다.")

            # --- 두 분석 사이에 구분선 추가 ---
            st.divider()

            # --- 두 번째 분석: 카테고리별 리텐션 ---
            st.subheader("🗂️ 카테고리별 고객 리텐션")
            
            # 카테고리 선택 필터 (너비 조절을 위해 컬럼 사용)
            filter_col, _ = st.columns([1, 2])
            with filter_col:
                available_categories = sorted(products_master['category'].unique())
                selected_category = st.selectbox(
                    "분석할 카테고리를 선택하세요:", 
                    available_categories,
                    index=available_categories.index('Intimates') if 'Intimates' in available_categories else 0
                )
            
            # 카테고리별 리텐션 차트 생성 및 표시
            cat_heatmap_fig, cat_cohort_df = create_category_cohort_heatmap(
                order_items_master, 
                products_master,
                selected_category,
                show_annotations
            )
            if cat_heatmap_fig:
                st.pyplot(cat_heatmap_fig)
                with st.expander("상세 데이터 보기"):
                    st.dataframe(cat_cohort_df.style.format("{:.2%}"))
            else:
                st.warning(f"'{selected_category}' 카테고리에 해당하는 데이터가 없습니다.")
        

    


    with tab2:
        st.subheader("월별 첫 구매 고객 재구매율 분석")

        filter_col, _ = st.columns([1, 2])
        with filter_col:
            # 2. st.slider를 사용하여 필터를 메인 페이지에 배치합니다.
            max_age_option = st.slider(
                "최대 경과 개월 수:", 
                min_value=1, 
                max_value=12, 
                value=12, 
                help="히트맵에 표시할 최대 재구매 경과 개월 수를 선택합니다."
            )
        
        # 고급 코호트 분석 함수 호출
        cohort_fig, cohort_df = create_advanced_cohort_heatmap(
            order_items_master, 
            max_age_option, 
            show_annotations
        )

        if cohort_fig:
            st.pyplot(cohort_fig)
            with st.expander("상세 데이터 보기"):
                # .style.format()은 PeriodIndex에서 오류가 발생할 수 있으므로 안전하게 처리
                try:
                    st.dataframe(cohort_df.style.format("{:.2%}"))
                except Exception:
                    st.dataframe(cohort_df)
        else:
            # 함수 내부에서 st.warning으로 이미 메시지를 보여주므로 여기서는 pass
            pass

                # ✨ 수정: 함수 호출 시 year 인자 제거
        repeat_fig, repeat_df = create_repeat_purchaser_chart(order_items_master)

        if repeat_fig:
            st.pyplot(repeat_fig)
            with st.expander("상세 데이터 보기"):
                st.dataframe(repeat_df[['returning_users','purchasers','repeat_purchaser_rate']])
        else:
            # 함수 내부에서 이미 경고 메시지를 표시함
            pass

        st.divider()

        st.subheader("주간 코호트 재구매율")



        # --- 필터 위젯 ---
        # 데이터에서 선택 가능한 월 목록 동적 생성
        temp_df = order_items_master.copy()
        temp_df['cohort_month'] = pd.to_datetime(temp_df['created_at']).dt.to_period('M').astype(str)
        available_months = sorted(temp_df['cohort_month'].unique(), reverse=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            selected_month = st.selectbox("분석할 코호트 월 선택:", available_months)
        with col2:
            week_options = ['All'] + list(range(1, 6))
            selected_week = st.selectbox("주차 필터 (Wn):", week_options, help="해당 월의 n번째 주에 시작된 코호트만 필터링합니다.")

        col_slider, _ = st.columns([2, 1])
        with col_slider:
            max_age_option = st.slider("최대 경과 주 수:", 1, 52, 12)

        show_annotations = st.checkbox("히트맵에 값(%) 표시", value=True)
        st.divider()

        # --- 차트 생성 및 표시 ---
        if selected_month:
            weekly_fig, weekly_df = create_weekly_cohort_heatmap(
                order_items_master, 
                selected_month,
                selected_week,
                max_age_option, 
                show_annotations
            )

            if weekly_fig:
                st.pyplot(weekly_fig)
                with st.expander("상세 데이터 보기"):
                    st.dataframe(weekly_df.style.format("{:.2%}"))
            else:
                # 함수 내부에서 이미 경고 메시지를 표시함
                pass

        st.divider()
        st.subheader("일별 코호트 재구매율")

        # --- 필터 위젯 ---
        # 데이터에서 선택 가능한 월 목록 생성
        temp_df = order_items_master.copy()
        temp_df['cohort_month'] = pd.to_datetime(temp_df['created_at']).dt.to_period('M').astype(str)
        available_months = sorted(temp_df['cohort_month'].unique(), reverse=True)
        
        temp_df = order_items_master.copy()
        temp_df['cohort_month'] = pd.to_datetime(temp_df['created_at']).dt.to_period('M').astype(str)
        available_months = sorted(temp_df['cohort_month'].unique(), reverse=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            selected_month2 = st.selectbox("분석할 코호트 월 선택:", available_months,key='source_selectbox_tab2')
        with col2:
            # 월~W5, 전체(All)
            week_options = ['All'] + [i for i in range(1, 6)]
            selected_week = st.selectbox("주차 필터 (Wn):", week_options)

        col_slider, _ = st.columns([2, 1])
        with col_slider:
            max_age_option = st.slider("최대 경과 일 수:", 1, 60, 30)

        st.divider()

        # --- 차트 생성 및 표시 ---
        if selected_month:
            daily_fig, daily_df = create_daily_cohort_heatmap(
                order_items_master, 
                selected_month2,
                selected_week,
                max_age_option, 
                show_annotations
            )

            if daily_fig:
                st.pyplot(daily_fig)
                with st.expander("상세 데이터 보기"):
                    st.dataframe(daily_df.style.format("{:.2%}"))
            else:
                # 함수 내부에서 이미 경고 메시지를 표시함
                pass

        

            

    
    with tab3:
        st.subheader("요일에 따른 재구매 패턴 심층 분석")
        st.info("첫 구매 요일 또는 실제 재구매가 발생한 요일을 기준으로 재구매 건수와 비율의 차이를 분석합니다.")

        # 새로 만든 함수 호출
        weekday_fig, order_data, cohort_data = create_weekday_repeat_purchase_charts(order_items_master, start_date, end_date)

        if weekday_fig:
            st.pyplot(weekday_fig)
            
            with st.expander("상세 데이터 보기"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### 재구매 발생 요일 기준")
                    st.dataframe(order_data[['Weekday', 'Repeat_Orders', 'Exposure', 'Repeat_Rate']].style.format({'Repeat_Rate': '{:.2%}'}))
                with col2:
                    st.write("#### 첫 구매 요일 기준")
                    st.dataframe(cohort_data[['Weekday', 'Repeat_Orders', 'Exposure', 'Repeat_Rate']].style.format({'Repeat_Rate': '{:.2%}'}))
        else:
            # 함수 내부에서 이미 경고 메시지를 표시함
            pass

        st.info("첫 구매 요일 또는 실제 재구매가 발생한 요일을 기준으로 재구매 건수와 비율의 차이를 분석합니다.")
        
        # --- ✨ [섹션 추가] 주중/주말 재구매율 비교 ---
        weekday_fig, weekday_tbl = create_weekday_weekend_chart(order_items_master, start_date, end_date)
        
        if weekday_fig:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("#### 분석 요약 테이블")
                st.dataframe(weekday_tbl, hide_index=True)
            with col2:
                st.pyplot(weekday_fig)
        else:
            st.warning("주중/주말 분석을 위한 데이터가 부족합니다.")


    
    