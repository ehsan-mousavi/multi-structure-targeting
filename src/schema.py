#!/usr/bin/env python2
# -*- coding: utf-8 -*-


CONTINUOUS_COLS = [

    # t10d features
    't10d_n_eats_orders',
    't10d_sum_eats_g1g1_promo_spend',
    't10d_n_rides_trips',

    # t30d features
    't30d_gb_sum',
    't30d_ni_sum',
    't30d_gb_per_order',
    't30d_ni_per_order',
    't30d_n_eats_orders',
    't30d_n_eats_redeemed_orders',
    't30d_sum_eats_promo_spend',
    't30d_n_rides_trips',

    # t90d features
    't90d_gb_sum',
    't90d_ni_sum',
    't90d_gb_per_order',
    't90d_ni_per_order',
    't90d_n_eats_orders',
    't90d_n_eats_redeemed_orders',
    't90d_sum_eats_promo_spend',
    't90d_n_rides_trips',
    't90d_avg_basket_size',
    't90d_avg_delivery_fee',
    't90d_prop_with_bandwagon',
    't90d_prop_with_baskets_below_10',
    't90d_prop_with_baskets_below_30',
    't90d_prop_with_delivery_fees_above_3',
    't90d_prop_with_resto_promo',
    't90d_tips_sum',
    't90d_avg_upfront_fare',
    't90d_days_since_last_eats_order',
    't90d_days_since_last_rides_trip',

    # marketing features
    't60d_marketing_emails_clicked',
    'is_high_frequency_eater',
    'is_inactive_28d_eater',
    'is_med_frequency_eater',
    'is_occasional_eater'

    ## ---- city features (deprecated)
    # 't30d_cf_avg_basket_size',
    # 't30d_cf_avg_delivery_fee',
    # 't30d_cf_gb_per_order',
    # 't30d_cf_gb_stddev',
    # 't30d_cf_gb_sum',
    # 't30d_cf_n_eats_orders',
    # 't30d_cf_ni_per_order',
    # 't30d_cf_ni_stddev',
    # 't30d_cf_ni_sum',
    # 't30d_cf_p50_basket_size',
    # 't30d_cf_p50_delivery_fee',
    # 't30d_cf_prop_with_bandwagon',
    # 't30d_cf_prop_with_eats_g1g1_promo',
    # 't30d_cf_prop_with_resto_promo',
    # 't30d_cf_prop_with_sof',
    # 't30d_cf_sum_eats_g1g1_promo_spend',
    # 't30d_cf_tips_per_order',
    # 't30d_cf_tips_sum',
    ## ---- city features (deprecated)

    ## ---- redundant features
    # 't30d_n_eats_promotions_used',
    # 't30d_sum_eats_g1g1_promo_spend'
    # 't60d_marketing_emails_caused_unsub',
    # 't60d_marketing_emails_click_rate',
    # 't60d_marketing_emails_delivered',
    # 't60d_marketing_emails_open_rate',
    # 't60d_marketing_emails_opened',
    # 't90d_n_eats_promotions_used',
    # 't90d_sum_eats_g1g1_promo_spend',

    ## ---- noisy features
    # 't30d_gb_per_order_citynorm',
    # 't30d_ni_per_order_citynorm',
    # 't90d_avg_basket_size_citynorm',
    # 't90d_avg_delivery_fee_citynorm',
    # 't90d_eats_g1g1_per_order_citynorm',
    # 't30d_gb_stddev',
    # 't30d_ni_stddev',
    # 't90d_gb_stddev',
    # 't90d_ni_stddev',
    # 't90d_prop_of_orders_with_surge',
    # 't90d_prop_of_rides_as_pool_trips',
    # 't90d_prop_of_rides_with_surge',
    # 't90d_eats_completion_rate',
    # 't90d_prop_with_delivery_fees_above_0',
    # 't90d_rides_completion_rate',

]

CATEGORICAL_COLS = [
    'city_id'
]

