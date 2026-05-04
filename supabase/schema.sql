create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

create table if not exists public.watchlist_items (
  user_id uuid not null references auth.users(id) on delete cascade,
  ticker text not null,
  exchange text not null default '',
  name text not null default '',
  symbol text not null,
  source text not null default '',
  enabled boolean not null default true,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  primary key (user_id, symbol)
);

create table if not exists public.analysis_snapshots (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  run_type text not null,
  title text not null,
  row_count integer not null default 0,
  summary jsonb not null default '{}'::jsonb,
  rows jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_watchlist_items_user_updated
on public.watchlist_items (user_id, updated_at desc);

create index if not exists idx_analysis_snapshots_user_created
on public.analysis_snapshots (user_id, created_at desc);

drop trigger if exists set_watchlist_items_updated_at on public.watchlist_items;
create trigger set_watchlist_items_updated_at
before update on public.watchlist_items
for each row
execute function public.set_updated_at();

drop trigger if exists set_analysis_snapshots_updated_at on public.analysis_snapshots;
create trigger set_analysis_snapshots_updated_at
before update on public.analysis_snapshots
for each row
execute function public.set_updated_at();

alter table public.watchlist_items enable row level security;
alter table public.analysis_snapshots enable row level security;

drop policy if exists "watchlist_items_select_own" on public.watchlist_items;
create policy "watchlist_items_select_own"
on public.watchlist_items
for select
using (auth.uid() = user_id);

drop policy if exists "watchlist_items_insert_own" on public.watchlist_items;
create policy "watchlist_items_insert_own"
on public.watchlist_items
for insert
with check (auth.uid() = user_id);

drop policy if exists "watchlist_items_update_own" on public.watchlist_items;
create policy "watchlist_items_update_own"
on public.watchlist_items
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "watchlist_items_delete_own" on public.watchlist_items;
create policy "watchlist_items_delete_own"
on public.watchlist_items
for delete
using (auth.uid() = user_id);

drop policy if exists "analysis_snapshots_select_own" on public.analysis_snapshots;
create policy "analysis_snapshots_select_own"
on public.analysis_snapshots
for select
using (auth.uid() = user_id);

drop policy if exists "analysis_snapshots_insert_own" on public.analysis_snapshots;
create policy "analysis_snapshots_insert_own"
on public.analysis_snapshots
for insert
with check (auth.uid() = user_id);

drop policy if exists "analysis_snapshots_update_own" on public.analysis_snapshots;
create policy "analysis_snapshots_update_own"
on public.analysis_snapshots
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "analysis_snapshots_delete_own" on public.analysis_snapshots;
create policy "analysis_snapshots_delete_own"
on public.analysis_snapshots
for delete
using (auth.uid() = user_id);
