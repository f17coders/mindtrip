package com.a303.notificationms.domain.memberToken.repository;

import com.a303.notificationms.domain.memberToken.MemberToken;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;

@RequiredArgsConstructor
@Slf4j
@Repository
public class MemberTokenRepositoryImpl implements MemberTokenCustomRepository{

    private final MongoTemplate mongoTemplate;

    @Override
    public MemberToken findByMemberId(int memberId) {

        Query query = new Query(
                Criteria.where("member_id").is(memberId)
        );

        return mongoTemplate.findOne(query, MemberToken.class);

    }
}
