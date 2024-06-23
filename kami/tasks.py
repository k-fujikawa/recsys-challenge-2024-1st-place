from invoke import task



@task
def create_candidates(ctx, debug=False):
    ctx.run("python preprocess/test_demo/run.py")
    scripts = [        
        "preprocess/make_candidate/run.py"
    ]
    for script in scripts:
        cmd = f"python {script}"
        if debug:
            cmd += " exp=small"
        else:
            cmd += " exp=large"
        print(cmd)
        ctx.run(cmd)

@task
def create_features(ctx, debug=False):
    scripts = [
        "features/ua_topics_sim_count_svd_feat/run.py",
        "features/a_base/run.py",
        "features/a_click_ranking/run.py",
        "features/a_additional_feature/run.py",
        "features/a_click_ratio/run.py",
        "features/a_click_ratio_multi/run.py",
        "features/i_base_feat/run.py",
        "features/i_stat_feat/run.py",
        "features/i_viewtime_diff/run.py",
        "features/i_article_stat_v2/run.py",
        "features/u_stat_history/run.py",
        "features/u_click_article_stat_v2/run.py",
        "features/y_transition_prob_from_first/run.py",
        "features/c_appear_imp_count_v7/run.py",
        "features/c_appear_imp_count_read_time_per_inview_v7/run.py",
        "features/c_topics_sim_count_svd/run.py",
        "features/c_title_tfidf_svd_sim/run.py",
        "features/c_subtitle_tfidf_svd_sim/run.py",
        "features/c_body_tfidf_svd_sim/run.py",
        "features/c_category_tfidf_sim/run.py",
        "features/c_subcategory_tfidf_sim/run.py",
        "features/c_entity_groups_tfidf_sim/run.py",
        "features/c_ner_clusters_tfidf_sim/run.py",
        "features/c_article_publish_time_v5/run.py",
        "features/c_is_already_clicked/run.py",
    ]

    for script in scripts:
        print("*"*20)
        cmd = f"python {script}"
        if debug:
            cmd += " exp=small"
        else:
            cmd += " exp=large"
        print(cmd)
        ctx.run(cmd)
        print()

@task
def create_datasets(ctx, debug=False):
    scripts = [
        "preprocess/dataset067/run.py",
    ]
    for script in scripts:
        print("*"*20)
        cmd = f"python {script}"
        if debug:
            cmd += " exp=small"
        else:
            cmd += " exp=large"
        print(cmd)
        ctx.run(cmd)
        print()

@task
def train(ctx, debug=False):
    scripts = [
        ("experiments/015_train_third/run.py", '067_001'),
        ("experiments/016_catboost/run.py", '067'),
    ]
    for script, exp in scripts:
        print("*"*20)
        if debug:
            cmd = f"python {script} exp=small{exp} debug=True"
        else:
            cmd = f"python {script} exp=large{exp} debug=True" # remove `debug=True` when you want to use wandb
        print(cmd)
        ctx.run(cmd)
        print()


@task(create_candidates, create_features, create_datasets, train)
def run_all(ctx, debug=False):
    pass